#include "reconstruction_worker.h"
#include "tilt_correction.h"
#include "fbp_reconstructor.h"
#include "gridrec_reconstructor.h"
#include "slice_reconstructor.h"
#include "ring_filter.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

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
    } else {
        for (int a = 0; a < params_.n_angles; ++a) {
            double frac = (params_.n_angles > 1) ? static_cast<double>(a) / (params_.n_angles - 1) : 0.0;
            double deg = params_.rotationStartDeg + params_.lastAngleDeg * frac;
            angles[a] = deg * M_PI / 180.0;
        }
    }
    // Rotating the output about the rotation axis is just a constant offset on every projection
    // angle: exact, with no interpolation of the reconstructed image. +offset turns the image
    // clockwise on screen (checked against FBP and Gridrec).
    const double offsetRad = params_.rotationDeg * M_PI / 180.0;
    for (double& a : angles)
        a += offsetRad;
    return angles;
}

void ReconstructionWorker::runPreview()
{
    try {
        int x, y, w, h;
        params_.roiRect.getRect(&x, &y, &w, &h);

        const int binning = std::max(1, params_.binning);
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

            if (params_.startStage == StartStage::CorrectedProjections && !proj_->hasCorrectedScan())
                throw std::runtime_error("Run Correct Scan first - no corrected projections in memory");

            for (int a = 0; a < params_.n_angles; ++a) {
                cv::Mat proj;
                if (params_.startStage == StartStage::CorrectedProjections) {
                    proj = proj_->correctedProjection(a).clone();
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
            if (params_.largeStripeEnabled)
                RingFilter::remove_large_stripes(sino, params_.largeStripeSnr, params_.largeStripeWindow);
            if (params_.smallStripeEnabled)
                RingFilter::remove_small_stripes(sino, params_.smallStripeWindow);
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
