#ifndef RECONSTRUCTION_WORKER_H
#define RECONSTRUCTION_WORKER_H
#include <QObject>
#include <QRect>
#include <QString>
#include <QMetaType>
#include <vector>
#include "proj_correction.h"
#include "fbp_reconstructor.h"
#include "gridrec_reconstructor.h"

// Runs the full low-RAM reconstruction pipeline (tilt-correct + stream
// sinograms to SSD, then read back sequentially: CoR shift -> ring filter ->
// FBP -> polar ring removal -> write slice) on a worker thread, reporting
// progress via signals.
// Only ever holds one projection or one sinogram/slice in memory at a time.
class ReconstructionWorker : public QObject
{
    Q_OBJECT

public:
    // Where to enter the pipeline. CorrectedProjections and Sinograms let the user re-run later
    // stages (e.g. to retune ring-filter/CoR settings) without redoing the expensive earlier ones.
    enum class StartStage
    {
        RawScan,               // scan/ -> flat-field + (-log) + tilt-correct + write sino/
        CorrectedProjections,  // corr/ (already flat-field + phase-retrieval corrected) -> tilt-correct + write sino/
        Sinograms              // sino/ already has tilt-corrected, log-transformed data -> skip straight to CoR/ring/FBP
    };

    // Which slice reconstructor to run in the CoR/ring-filter -> reconstruct tail. Gridrec is the
    // Fourier-domain (Dowd/Marone) method: same ramp+window filtering as Fbp (see fbpFilter below,
    // which configures both), but O(N^2 log N) gridding+2D-IFFT instead of Fbp's O(n_angles*N^2)
    // real-space backprojection - much faster at high angle counts, at the cost of a small amount
    // of gridding-kernel approximation error relative to Fbp's direct (exact-interpolation) result.
    // Available on both the Accelerate (macOS) and CUDA (Linux) builds, like Fbp itself.
    enum class ReconAlgorithm
    {
        Fbp,
        Gridrec
    };

    struct Params
    {
        QString workingPath;
        StartStage startStage = StartStage::RawScan;
        int n_angles = 0;
        double rotationStartDeg = 0.0;
        double lastAngleDeg = 180.0;
        // When set, buildAngles() uses these recorded rotation-encoder readings (degrees, one per
        // projection index, size must equal n_angles) instead of assuming an even angular step
        // from rotationStartDeg to lastAngleDeg. See angle_file_reader.h. Only affects the FBP
        // backprojection tail (Stage 2/3) - Stage 1 reads projections in index order regardless,
        // so this isn't part of PreviewCache's key.
        bool useAngleFile = false;
        std::vector<double> anglesDeg;
        QRect roiRect;
        double tiltDeg = 0.0;
        double corOffset = 0.0;
        bool ringEnabled = true;
        int ringLevel = 3;
        double ringSigma = 2.0;
        int ringOrder = 3;
        int ringPad = 200;
        int ringMaskInnerRadius = 0; // 0 = no user mask beyond remove_stripes' own built-in center
        int ringMaskOuterRadius = 0; // floor; see RingFilter::remove_stripes
        // The mask is the annulus [inner, outer) the wavelet filter skips. The polar filter (below)
        // then runs only in that same annulus: it stops at outer, and with inner > 0 it leaves the
        // disc inside inner untouched, since that disc is the wavelet filter's - see
        // PolarRingRemoval::centerExclusionRadius for why the center must not go to the polar filter.
        // Post-reconstruction, polar-domain ring removal (PolarRingRemoval::remove_ring, see
        // ring_removal_polar.h) - runs on the reconstructed slice itself, right after
        // reconstruct_slice(), as a complement to the sinogram-domain wavelet-Fourier filter above.
        // Named polarRing* (not ring*) to stay distinct from that filter's fields.
        bool polarRingEnabled = false;
        double polarRingThresh = 300.0;
        double polarRingThreshMax = 300.0;
        double polarRingThreshMin = -100.0;
        double polarRingThetaMinDeg = 30.0;
        int polarRingWidth = 30;
        bool polarRingWrapBoundary = true;
        // See ring_removal_polar_cuda.h - a from-scratch CUDA port of the same algorithm, to try
        // whether the GPU actually beats the (already multi-threaded) CPU path for this workload.
        bool polarRingUseGpu = false;
        ReconAlgorithm algorithm = ReconAlgorithm::Fbp;
        FbpFilterType fbpFilter = FbpFilterType::Hamming; // ramp/window filter type, used by both algorithms
        double circMaskRatio = 0.995;
        int chunkRows = 128;
        // 1/2/3 = NxN block-average downsampling. The actual resizing happens inside
        // proj_correction (Proj_correction::setBinning), applied right after each image is read
        // and cropped - so it also speeds up spot filtering, flat-fielding, and phase retrieval,
        // not just the stages below. Only meaningful for RawScan/CorrectedProjections; ignored
        // when starting from Sinograms, since sino/ is already at whatever resolution it was
        // written at. This value must be set on proj_correction (via setBinning) to match before
        // run()/runPreview() is called - it's only used here to size n_rows/n_cols. corOffset and
        // the ring-mask radii are pixel values the caller measures/finds at whatever binning is
        // currently active, so they need no separate rescaling here.
        int binning = 1;
    };

    // Caches the 3 preview sinograms (bottom/mid/top) as built by Stage 1 - tilt-corrected,
    // log-transformed if applicable, but NOT yet CoR-shifted or ring-filtered - along with the
    // Params fields that determine whether they're still valid. CoR offset and every ring-filter/
    // FBP-filter/circular-mask setting are deliberately excluded from the key: those only affect
    // the cheap CoR-shift -> ring-filter -> FBP tail of runPreview(), so changing only those can
    // reuse this cache and skip Stage 1 (the expensive part - reading every projection once)
    // entirely. Changing anything that changes what data goes INTO Stage 1 (tilt, ROI, binning,
    // start stage, angle count/range) invalidates it.
    struct PreviewCache
    {
        bool valid = false;
        QString workingPath;
        StartStage startStage = StartStage::RawScan;
        int n_angles = 0;
        double rotationStartDeg = 0.0;
        double lastAngleDeg = 0.0;
        QRect roiRect;
        double tiltDeg = 0.0;
        int binning = 1;
        cv::Mat sinoBottom, sinoMid, sinoTop;

        bool matches(const Params& p) const;
    };

    // proj_correction is not owned and must remain valid (and untouched by other
    // threads) for the duration of run(). inputCache is only consulted by runPreview(); pass a
    // default-constructed one (valid=false) to always rebuild, which is what run() effectively
    // does anyway since it doesn't use this cache at all.
    ReconstructionWorker(Proj_correction* proj_correction, Params params,
                          PreviewCache inputCache, QObject* parent = nullptr);

public slots:
    void run();
    // Reconstructs only 3 rows - at 10%, 50%, and 90% of the ROI's height (in that order), not
    // the very first/last row, since those often fall outside the sample - so the user can
    // sanity-check CoR/tilt and ring-filter settings without waiting for a full reconstruction.
    // When starting from Sinograms, this just reads those 3 rows back from sino/ (cheap); otherwise
    // it still reads every projection once (each row needs a sample from every angle) but skips
    // writing any sinograms to disk and skips FBP for every row but these 3.
    void runPreview();

signals:
    void progress(int percent, QString message);
    void finished();
    void failed(QString error);
    // cache is whatever PreviewCache was actually used to produce bottom/mid/top - either the
    // inputCache passed into the constructor (if it was valid and matched), or a freshly-built
    // one otherwise. Callers should hold onto it and pass it back into the next runPreview() call.
    void previewFinished(cv::Mat bottom, cv::Mat mid, cv::Mat top, PreviewCache cache);

private:
    Proj_correction* proj_;
    Params params_;
    PreviewCache inputCache_;
    std::vector<double> buildAngles() const;
};

Q_DECLARE_METATYPE(ReconstructionWorker::PreviewCache)

#endif // RECONSTRUCTION_WORKER_H
