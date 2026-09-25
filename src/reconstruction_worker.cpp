#include "reconstruction_worker.h"
#include "tilt_correction.h"
#include "sinogram_io.h"
#include "fbp_reconstructor.h"
#include "gridrec_reconstructor.h"
#include "slice_reconstructor.h"
#include "ring_filter.h"
#include "ring_removal_polar.h"
#if !defined(__APPLE__)
#include "ring_removal_polar_cuda.h"
#endif
#include <filesystem>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace {

std::unique_ptr<SliceReconstructor> make_reconstructor(int n_cols, const ReconstructionWorker::Params& p)
{
    switch (p.algorithm) {
    case ReconstructionWorker::ReconAlgorithm::Gridrec:
        return std::make_unique<GridrecReconstructor>(n_cols, p.fbpFilter);
    case ReconstructionWorker::ReconAlgorithm::Fbp:
    default:
        return std::make_unique<FbpReconstructor>(n_cols, p.fbpFilter);
    }
}

} // namespace

ReconstructionWorker::ReconstructionWorker(Proj_correction* proj_correction, Params params,
                                            PreviewCache inputCache, QObject* parent)
    : QObject(parent), proj_(proj_correction), params_(std::move(params)), inputCache_(std::move(inputCache))
{
}

bool ReconstructionWorker::PreviewCache::matches(const Params& p) const
{
    // tiltDeg needs a tolerance, not exact equality: doubleSpinBox_tilt rounds to 4 decimals
    // (see mainwindow.ui), so a value that round-trips through it (e.g. because editing the CoR
    // offset spinbox switches both fields from "Automatic" to "Manual value" at once, since they
    // share one radio-button pair) can differ from the original full-precision double by up to
    // 0.00005 without the user having changed anything. That's far below any tilt-angle change
    // that could plausibly matter (a real edit via the spinbox moves it by at least 0.0001), so
    // it's safe to treat as "unchanged" and reuse the cache instead of re-reading every projection.
    constexpr double kTiltToleranceDeg = 7.5e-5;

    return valid
        && workingPath == p.workingPath
        && startStage == p.startStage
        && n_angles == p.n_angles
        && rotationStartDeg == p.rotationStartDeg
        && lastAngleDeg == p.lastAngleDeg
        && roiRect == p.roiRect
        && std::abs(tiltDeg - p.tiltDeg) <= kTiltToleranceDeg
        && binning == p.binning;
}

std::vector<double> ReconstructionWorker::buildAngles() const
{
    std::vector<double> angles(params_.n_angles);
    if (params_.useAngleFile) {
        if (static_cast<int>(params_.anglesDeg.size()) != params_.n_angles)
            throw std::runtime_error("Angle file has " + std::to_string(params_.anglesDeg.size()) +
                                      " entries but reconstruction is configured for " +
                                      std::to_string(params_.n_angles) + " angles");
        for (int a = 0; a < params_.n_angles; ++a)
            angles[a] = params_.anglesDeg[a] * M_PI / 180.0;
        return angles;
    }
    for (int a = 0; a < params_.n_angles; ++a) {
        double frac = (params_.n_angles > 1) ? static_cast<double>(a) / (params_.n_angles - 1) : 0.0;
        double deg = params_.rotationStartDeg + params_.lastAngleDeg * frac;
        angles[a] = deg * M_PI / 180.0;
    }
    return angles;
}

void ReconstructionWorker::run()
{
    try {
        int x, y, w, h;
        params_.roiRect.getRect(&x, &y, &w, &h);

        // Binning only applies to stages this run actually executes (RawScan/CorrectedProjections);
        // sino/ already has whatever resolution it was written at when starting from Sinograms.
        const int binning = (params_.startStage == StartStage::Sinograms)
                                 ? 1
                                 : std::max(1, params_.binning);
        const int n_rows = h / binning; // vertical extent -> number of reconstructed slices
        const int n_cols = w / binning; // detector width

        // --- Stage 1: tilt-correct each projection once, stream sinograms to SSD in chunks. ---
        // Skipped entirely when starting from existing sinograms.
        QString sinoDir = params_.workingPath + "/sino/";

        if (params_.startStage != StartStage::Sinograms) {
            SinogramWriter writer(sinoDir, n_cols);

            for (int rowStart = 0; rowStart < n_rows; rowStart += params_.chunkRows) {
                int numRows = std::min(params_.chunkRows, n_rows - rowStart);
                writer.beginChunk(rowStart, numRows, params_.n_angles);

                for (int a = 0; a < params_.n_angles; ++a) {
                    cv::Mat proj;
                    if (params_.startStage == StartStage::CorrectedProjections) {
                        // corr/ was already flat-field + phase-retrieval corrected by run_scan();
                        // that already produced the line-integral-like quantity FBP needs.
                        proj = proj_->get_projection_from_corr(a);
                    } else {
                        proj = proj_->get_projection_corr(a);
                        // get_projection_corr returns flat-corrected transmission T = (I-dark)/(flat-dark);
                        // FBP needs the Beer-Lambert line integral p = -ln(T). Clamp away from zero first
                        // since noise/spot-correction can leave T <= 0 at a few pixels.
                        cv::max(proj, 1e-6f, proj);
                        cv::log(proj, proj);
                        proj *= -1.0;
                    }
                    // proj_ (Proj_correction) already applies binning itself right after reading
                    // each image, so proj is already at (n_cols, n_rows) here - no resize needed.
                    TiltCorrection::apply(proj, params_.tiltDeg);
                    cv::Mat rowsBlock = proj.rowRange(rowStart, rowStart + numRows);
                    writer.addProjectionRows(a, rowsBlock);
                }
                writer.finishChunk();

                int pct = static_cast<int>(50.0 * (rowStart + numRows) / n_rows);
                emit progress(pct, QString("Writing sinograms: %1/%2").arg(rowStart + numRows).arg(n_rows));
            }
        } else {
            emit progress(50, QString("Using existing sinograms in %1").arg(sinoDir));
        }

        // --- Stage 2/3: read each sinogram back, shift to CoR, ring-filter, FBP, write slice. ---
        SinogramReader reader(sinoDir, n_rows);
        std::unique_ptr<SliceReconstructor> recon = make_reconstructor(n_cols, params_);
        std::vector<double> angles = buildAngles();

        // Radius (as a fraction of n_cols/2) the polar ring filter processes out to. Defaults to
        // the circular mask's own radius (circMaskRatio) - the annulus beyond that is already
        // zeroed by reconstruct_slice(), so there's nothing there to correct regardless. If the
        // wavelet filter's mask is active, its outer radius is used instead - a deliberate choice
        // (not just a performance cap) to only run the polar filter within that region.
        double polarRingMaskRatio = params_.circMaskRatio;
        if (params_.ringEnabled && params_.ringMaskOuterRadius > params_.ringMaskInnerRadius
            && params_.ringMaskOuterRadius > 0)
            polarRingMaskRatio = params_.ringMaskOuterRadius / (n_cols / 2.0);
        // ...and the mask's inner radius is where it starts: the disc inside it is left to the
        // wavelet filter, so the polar filter must not touch it (see centerExclusionRadius).
        const int polarCenterExclusion = PolarRingRemoval::centerExclusionRadius(
            params_.ringEnabled, params_.ringMaskInnerRadius, params_.ringMaskOuterRadius);

#if !defined(__APPLE__)
        // reconstruct_slice() always produces a square n_cols x n_cols slice, so that's the fixed
        // size to construct the GPU backend's cached buffers for. Constructed once here (like recon
        // above) and shared across the row pool below - PolarRingCudaBackend serializes its own
        // calls internally, so no extra locking is needed at this call site.
        std::unique_ptr<PolarRingCudaBackend> polarRingGpu;
        if (params_.polarRingEnabled && params_.polarRingUseGpu)
            polarRingGpu = std::make_unique<PolarRingCudaBackend>(n_cols, n_cols, polarRingMaskRatio);
#endif

        QString recoDir = params_.workingPath + "/reco/";
        fs::create_directories(recoDir.toStdString());

        // corOffset and the ring-mask radii are pixel measurements the user takes by eye (Find
        // CoR/Tilt, or reading a radius off a reconstructed slice) at whatever binning was active
        // at the time - MainWindow keeps proj_correction's binning in sync when each of those
        // happens, so these are already in the right pixel domain here and need no rescaling.

        // Rows are fully independent (each reads its own sinogram slice and writes its own
        // output file), so fan this out across a small thread pool instead of doing it serially.
        std::atomic<int> nextRow{0};
        std::atomic<int> completedRows{0};
        std::atomic<bool> failed{false};
        std::mutex errorMutex;
        QString firstError;

        unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
        numThreads = std::min(numThreads, static_cast<unsigned int>(std::max(1, n_rows)));

        auto reconstructRows = [&]() {
            for (;;) {
                int row = nextRow.fetch_add(1);
                if (row >= n_rows || failed.load(std::memory_order_relaxed))
                    return;
                try {
                    cv::Mat sino = reader.readSlice(row);

                    // Shift the sinogram so the rotation axis lands at the detector center.
                    if (params_.corOffset != 0.0)
                        recon->shift_sinogram(sino, params_.corOffset);

                    if (params_.ringEnabled)
                        RingFilter::remove_stripes(sino, params_.ringLevel, params_.ringSigma,
                                                    params_.ringOrder, params_.ringPad,
                                                    params_.ringMaskInnerRadius, params_.ringMaskOuterRadius);

                    cv::Mat slice = recon->reconstruct_slice(sino, angles, params_.circMaskRatio);

                    // Post-reconstruction, polar-domain ring removal - complements the
                    // sinogram-domain wavelet filter above. CPU path uses parallel=false: this
                    // lambda already runs inside its own per-row thread pool below, and nesting
                    // remove_ring's own internal thread pool inside that would oversubscribe the
                    // machine (the same bug already fixed once in post_process_worker.cpp). The
                    // GPU path needs no such flag - PolarRingCudaBackend serializes itself.
                    if (params_.polarRingEnabled) {
                        cv::Mat beforePolar;
                        if (polarCenterExclusion > 0)
                            beforePolar = slice.clone();
#if !defined(__APPLE__)
                        if (polarRingGpu) {
                            slice = polarRingGpu->remove_ring(slice, params_.polarRingThresh,
                                                               params_.polarRingThreshMax, params_.polarRingThreshMin,
                                                               params_.polarRingThetaMinDeg, params_.polarRingWidth,
                                                               params_.polarRingWrapBoundary);
                        } else
#endif
                        {
                            // macOS has no CUDA polar-ring backend (see the __APPLE__ guards above
                            // and in the includes/PolarRingCudaBackend construction) - always the
                            // CPU path there, regardless of params_.polarRingUseGpu.
                            slice = PolarRingRemoval::remove_ring(slice, params_.polarRingThresh,
                                                                   params_.polarRingThreshMax, params_.polarRingThreshMin,
                                                                   params_.polarRingThetaMinDeg, params_.polarRingWidth,
                                                                   params_.polarRingWrapBoundary, /*parallel=*/false,
                                                                   polarRingMaskRatio);
                        }
                        PolarRingRemoval::restore_center(beforePolar, slice, polarCenterExclusion);
                    }

                    QString outPath = recoDir + QString("reco_%1.tiff").arg(row, 5, 10, QChar('0'));
                    cv::imwrite(outPath.toStdString(), slice);

                    int done = completedRows.fetch_add(1) + 1;
                    int pct = 50 + static_cast<int>(50.0 * done / n_rows);
                    emit progress(pct, QString("Reconstructing: %1/%2").arg(done).arg(n_rows));
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (!failed.exchange(true))
                        firstError = QString::fromStdString(e.what());
                    return;
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(numThreads);
        for (unsigned int t = 0; t < numThreads; ++t)
            pool.emplace_back(reconstructRows);
        for (auto& th : pool)
            th.join();

        if (failed.load())
            throw std::runtime_error(firstError.toStdString());

        emit finished();
    } catch (const std::exception& e) {
        emit failed(QString::fromStdString(e.what()));
    }
}

void ReconstructionWorker::runPreview()
{
    try {
        int x, y, w, h;
        params_.roiRect.getRect(&x, &y, &w, &h);

        const int binning = (params_.startStage == StartStage::Sinograms)
                                 ? 1
                                 : std::max(1, params_.binning);
        const int n_rows = h / binning;
        const int n_cols = w / binning;
        if (n_rows <= 0 || n_cols <= 0)
            throw std::runtime_error("ReconstructionWorker::runPreview: ROI is empty");

        // 10%, 50%, and 90% of the way up the ROI, in that order - not the very first/last row,
        // since those often fall outside the sample (empty air) in practice.
        const int targetRows[3] = {
            std::clamp(static_cast<int>(std::lround(n_rows * 0.10)), 0, n_rows - 1),
            n_rows / 2,
            std::clamp(static_cast<int>(std::lround(n_rows * 0.90)), 0, n_rows - 1)
        };

        std::array<cv::Mat, 3> sinos;
        PreviewCache outputCache;

        if (inputCache_.matches(params_)) {
            // Nothing that affects the sinogram itself has changed since last time (only CoR
            // offset and/or ring-filter/FBP-filter/circular-mask settings, all applied below) -
            // reuse it and skip straight to the cheap tail. Clone since shift_sinogram and the
            // ring filters modify in place, and the cache may be reused again unchanged later.
            sinos[0] = inputCache_.sinoBottom.clone();
            sinos[1] = inputCache_.sinoMid.clone();
            sinos[2] = inputCache_.sinoTop.clone();
            outputCache = inputCache_;
            emit progress(70, QString("Preview: reusing cached sinograms (only CoR/ring-filter settings changed)"));
        } else {
            for (auto& s : sinos)
                s = cv::Mat::zeros(params_.n_angles, n_cols, CV_32FC1);

            QString sinoDir = params_.workingPath + "/sino/";

            if (params_.startStage == StartStage::Sinograms) {
                SinogramReader reader(sinoDir, n_rows);
                for (int k = 0; k < 3; ++k)
                    sinos[k] = reader.readSlice(targetRows[k]).clone();
                emit progress(70, QString("Preview: read cached sinograms"));
            } else {
                for (int a = 0; a < params_.n_angles; ++a) {
                    cv::Mat proj;
                    if (params_.startStage == StartStage::CorrectedProjections) {
                        proj = proj_->get_projection_from_corr(a);
                    } else {
                        proj = proj_->get_projection_corr(a);
                        cv::max(proj, 1e-6f, proj);
                        cv::log(proj, proj);
                        proj *= -1.0;
                    }
                    TiltCorrection::apply(proj, params_.tiltDeg);

                    for (int k = 0; k < 3; ++k)
                        proj.row(targetRows[k]).copyTo(sinos[k].row(a));

                    if (a % 20 == 0 || a == params_.n_angles - 1) {
                        int pct = static_cast<int>(70.0 * (a + 1) / params_.n_angles);
                        emit progress(pct, QString("Preview: reading projection %1/%2").arg(a + 1).arg(params_.n_angles));
                    }
                }
            }

            outputCache.valid = true;
            outputCache.workingPath = params_.workingPath;
            outputCache.startStage = params_.startStage;
            outputCache.n_angles = params_.n_angles;
            outputCache.rotationStartDeg = params_.rotationStartDeg;
            outputCache.lastAngleDeg = params_.lastAngleDeg;
            outputCache.roiRect = params_.roiRect;
            outputCache.tiltDeg = params_.tiltDeg;
            outputCache.binning = params_.binning;
            // Deep-clone, not shallow-assign: cv::Mat assignment shares the underlying pixel
            // buffer, and the shift/ring-filter loop below mutates sinos[k] in place (via the
            // aliased local `sino`). Without cloning here, the "pristine pre-filter" cache would
            // silently end up holding whatever filter was applied this run, corrupting every
            // future cache hit with this run's settings baked in on top of its own.
            outputCache.sinoBottom = sinos[0].clone();
            outputCache.sinoMid = sinos[1].clone();
            outputCache.sinoTop = sinos[2].clone();
        }

        std::unique_ptr<SliceReconstructor> recon = make_reconstructor(n_cols, params_);
        std::vector<double> angles = buildAngles();
        std::array<cv::Mat, 3> slices;

        for (int k = 0; k < 3; ++k) {
            cv::Mat sino = sinos[k];
            if (params_.corOffset != 0.0)
                recon->shift_sinogram(sino, params_.corOffset);
            if (params_.ringEnabled)
                RingFilter::remove_stripes(sino, params_.ringLevel, params_.ringSigma,
                                            params_.ringOrder, params_.ringPad,
                                            params_.ringMaskInnerRadius, params_.ringMaskOuterRadius);
            // Polar-domain ring removal is deliberately NOT applied here: mainwindow.cpp's
            // display_preview_slice() already applies it live (from ui->checkBox_ringRemovalEnable
            // etc.) on top of this cached raw-FBP preview slice, every time it redraws - that's
            // what lets the user retune the polar-ring parameters interactively without re-running
            // this (expensive) preview reconstruction. Baking it in here too would double-apply it.
            slices[k] = recon->reconstruct_slice(sino, angles, params_.circMaskRatio);

            int pct = 70 + (k + 1) * 10;
            emit progress(pct, QString("Preview: reconstructed slice %1/3").arg(k + 1));
        }

        emit previewFinished(slices[0], slices[1], slices[2], outputCache);
    } catch (const std::exception& e) {
        emit failed(QString("Preview failed: ") + e.what());
    }
}
