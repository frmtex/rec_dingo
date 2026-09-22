#include "inmemory_pipeline_worker.h"
#include "tilt_correction.h"
#include "fbp_reconstructor.h"
#include "ring_filter.h"
#include "ring_removal_polar.h"
#include "ring_removal_polar_cuda.h"
#include <filesystem>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

InMemoryPipelineWorker::InMemoryPipelineWorker(Proj_correction* proj_correction, ReconstructionWorker::Params params,
                                                QObject* parent)
    : QObject(parent), proj_(proj_correction), params_(std::move(params))
{
}

std::vector<double> InMemoryPipelineWorker::buildAngles() const
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

void InMemoryPipelineWorker::run()
{
    try {
        if (params_.startStage == ReconstructionWorker::StartStage::Sinograms)
            throw std::runtime_error("In-memory pipeline has no on-disk sinograms to resume from - "
                                      "pick Raw scan or Corrected projections as the start stage");

        int x, y, w, h;
        params_.roiRect.getRect(&x, &y, &w, &h);
        const int binning = std::max(1, params_.binning);
        const int n_rows = h / binning;
        const int n_cols = w / binning;

        // --- Stage 1: read + correct every projection, tilt-correct, keep entirely in RAM
        // (no corr/ written) - mirrors ReconstructionWorker::run()'s Stage 1 branch exactly:
        // StartStage::RawScan is flat-field + -log only (no phase retrieval, matching
        // get_projection_corr()'s use there - run_scan()'s corr/ output is a DIFFERENT, richer
        // correction that a plain "raw scan" reconstruction never applied even on the disk path);
        // StartStage::CorrectedProjections gets the full correction (spot filter + flat field +
        // intensity correction + phase retrieval) via get_projection_corrected_full(), computed
        // fresh from scan/ since there's no on-disk corr/ to read back in this mode.
        std::vector<cv::Mat> correctedProjections(params_.n_angles);
        for (int a = 0; a < params_.n_angles; ++a) {
            cv::Mat proj;
            if (params_.startStage == ReconstructionWorker::StartStage::CorrectedProjections) {
                proj = proj_->get_projection_corrected_full(a);
            } else {
                proj = proj_->get_projection_corr(a);
                cv::max(proj, 1e-6f, proj);
                cv::log(proj, proj);
                proj *= -1.0;
            }
            TiltCorrection::apply(proj, params_.tiltDeg);
            correctedProjections[a] = proj;

            if (a % 20 == 0 || a == params_.n_angles - 1) {
                int pct = static_cast<int>(33.0 * (a + 1) / params_.n_angles);
                emit progress(pct, QString("Correcting projections (in RAM): %1/%2").arg(a + 1).arg(params_.n_angles));
            }
        }

        // --- Stage 2: build sinograms directly from the in-RAM projection stack (no sino/
        // written) - same row-extraction SinogramWriter/Stage 1 does on disk, just indexing into
        // RAM instead of reading chunks back off disk.
        std::vector<cv::Mat> sinograms(n_rows);
        for (int r = 0; r < n_rows; ++r) {
            cv::Mat sino(params_.n_angles, n_cols, CV_32FC1);
            for (int a = 0; a < params_.n_angles; ++a)
                correctedProjections[a].row(r).copyTo(sino.row(a));
            sinograms[r] = sino;

            if (r % 50 == 0 || r == n_rows - 1) {
                int pct = 33 + static_cast<int>(17.0 * (r + 1) / n_rows);
                emit progress(pct, QString("Building sinograms (in RAM): %1/%2").arg(r + 1).arg(n_rows));
            }
        }
        correctedProjections.clear(); // no longer needed - free before the reconstruction pass below

        // --- Stage 3: shift to CoR, ring-filter, FBP, polar ring removal, write reco/ - identical
        // logic and thread-pool pattern to ReconstructionWorker::run()'s Stage 2/3, reading each
        // row's sinogram out of the in-RAM vector instead of a SinogramReader. Different threads
        // claim different (unique) row indices, so concurrent access to different sinograms[row]
        // elements is safe - the same guarantee SinogramReader's per-file reads gave, just RAM-backed.
        FbpReconstructor fbp(n_cols, params_.fbpFilter);
        std::vector<double> angles = buildAngles();

        double polarRingMaskRatio = params_.circMaskRatio;
        if (params_.ringEnabled && params_.ringMaskOuterRadius > params_.ringMaskInnerRadius
            && params_.ringMaskOuterRadius > 0)
            polarRingMaskRatio = params_.ringMaskOuterRadius / (n_cols / 2.0);

        std::unique_ptr<PolarRingCudaBackend> polarRingGpu;
        if (params_.polarRingEnabled && params_.polarRingUseGpu)
            polarRingGpu = std::make_unique<PolarRingCudaBackend>(n_cols, n_cols, polarRingMaskRatio);

        QString recoDir = params_.workingPath + "/reco/";
        fs::create_directories(recoDir.toStdString());

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
                    cv::Mat sino = sinograms[row];

                    if (params_.corOffset != 0.0)
                        fbp.shift_sinogram(sino, params_.corOffset);

                    if (params_.ringEnabled)
                        RingFilter::remove_stripes(sino, params_.ringLevel, params_.ringSigma,
                                                    params_.ringOrder, params_.ringPad,
                                                    params_.ringMaskInnerRadius, params_.ringMaskOuterRadius);

                    cv::Mat slice = fbp.reconstruct_slice(sino, angles, params_.circMaskRatio);

                    if (params_.polarRingEnabled) {
                        if (polarRingGpu) {
                            slice = polarRingGpu->remove_ring(slice, params_.polarRingThresh,
                                                               params_.polarRingThreshMax, params_.polarRingThreshMin,
                                                               params_.polarRingThetaMinDeg, params_.polarRingWidth,
                                                               params_.polarRingWrapBoundary);
                        } else {
                            slice = PolarRingRemoval::remove_ring(slice, params_.polarRingThresh,
                                                                   params_.polarRingThreshMax, params_.polarRingThreshMin,
                                                                   params_.polarRingThetaMinDeg, params_.polarRingWidth,
                                                                   params_.polarRingWrapBoundary, /*parallel=*/false,
                                                                   polarRingMaskRatio);
                        }
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
