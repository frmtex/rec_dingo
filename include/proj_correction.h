#ifndef PROJ_CORRECTION_H
#define PROJ_CORRECTION_H
#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include <vector>
#include <mutex>
#include "distortion_correction.h"
#if !defined(__APPLE__)
#include "spot_filter_cuda.h"
#include "phase_retrieval_cuda.h"
#endif

class Proj_correction
{
public:
    Proj_correction()
    {
    }
    void setData(QString _data_path, QRect _roi_rect)
    {
        data_path = _data_path;
        roi_rect = _roi_rect;
        clearCorrectedScan();
    }
    // Spot (hot/dead pixel) filter window size, applied in load_op_di/get_projection_corr/correctScan.
    // Guarded: correctScan()'s cache is only invalidated when the value actually changes, since
    // callers (buildReconstructionParams(), slot_find_cor()) re-apply the current UI value
    // defensively on every Preview/Reconstruction click.
    void setSpotKernelSize(int k) { if (k != spot_kernel_size) { spot_kernel_size = k; clearCorrectedScan(); } }
    // NxN block-average downsampling (1 = disabled), applied immediately after each image is
    // read and cropped to the ROI - before spot filtering, flat-fielding, or phase retrieval - so
    // every later step in this class runs on the smaller image too. Guarded, same reasoning as
    // setSpotKernelSize above.
    void setBinning(int b) { if (b != binning) { binning = b; clearCorrectedScan(); } }
    // Runs spot filtering and phase retrieval on the GPU (SpotFilterCudaBackend/
    // PhaseRetrievalCudaBackend) instead of the CPU (Spot_filter/phase_retrieval) - see
    // applySpotCorrection/applyPhaseRetrieval. The GPU backends are (re)built lazily, on first use
    // or whenever the image size changes. No CUDA build on macOS (see the __APPLE__ guards below),
    // so there this is a harmless no-op and applySpotCorrection/applyPhaseRetrieval always take the
    // CPU path regardless of the flag - same pattern as ReconstructionWorker's polar ring GPU path.
    // Guarded, same reasoning as setSpotKernelSize above.
    void setUseGpu(bool useGpu) { if (useGpu != use_gpu) { use_gpu = useGpu; clearCorrectedScan(); } }
    // Region of the corrected (cropped/binned) image containing only the primary beam, no sample -
    // used to correct for beam-intensity fluctuation over the course of a scan. Immediately reads
    // and corrects projection 0 to establish the reference mean; every projection returned by
    // get_projection_corr() and correctScan() afterwards (including index 0 itself) is rescaled so
    // its own mean in this region matches that reference. Coordinates are in the same pixel space
    // as im_show / get_projection_corr()'s output (post-crop, post-binning) - re-pick after
    // changing binning. Throws std::runtime_error if roi is empty or projection 0 can't be read.
    void setIntensityRoi(const QRect& roi);
    // Geometric distortion correction (see distortion_correction.h): when set, every frame -
    // projections, flats and darks alike - is read from a slightly larger area than the ROI and,
    // after spot filtering and flat-fielding, resampled onto the undistorted pixel grid, so the
    // corrected projection covers exactly the ROI (in corrected-image coordinates) as before. Spot
    // filtering deliberately runs before the resampling so single hot pixels aren't smeared over
    // several pixels first. Set before load_op_di(); pass nullptr to switch it off.
    void setDistortionCorrection(std::shared_ptr<const DistortionCorrection> correction)
    {
        distortion_ = std::move(correction);
        distortion_roi_valid = false;
        clearCorrectedScan(); // anything cached by correctScan() has the old geometry
    }
    void load_op_di();
    void get_first_image_corr();
    // Computes the fully-corrected (spot filter + flat field + intensity correction + phase
    // retrieval) projection for every angle - the same per-projection work
    // get_projection_corrected_full() does - and keeps the results in RAM (correctedProjection())
    // instead of writing them to disk. Deliberately NOT tilt-corrected: tilt is retuned
    // interactively afterward (via Preview), and baking it in here would force a re-run of this
    // whole pass on every tilt tweak - callers apply TiltCorrection::apply() themselves, fresh,
    // each time they read from the cache. progressCallback, if given, is invoked as
    // (doneCount, totalCount) after each projection is corrected - intended for a caller on
    // another thread to translate into a Qt progress signal.
    void correctScan(const std::function<void(int, int)>& progressCallback = nullptr);
    bool hasCorrectedScan() const { return !correctedProjections_.empty(); }
    int correctedScanCount() const { return static_cast<int>(correctedProjections_.size()); }
    const cv::Mat& correctedProjection(int index) const { return correctedProjections_.at(static_cast<size_t>(index)); }
    // Called once a reconstruction run has consumed correctScan()'s output, so it doesn't sit in
    // RAM after it's no longer needed. Also called automatically whenever a setting that would
    // change what correctScan() computes changes (see the setters above).
    void clearCorrectedScan() { correctedProjections_.clear(); correctedProjections_.shrink_to_fit(); }
    // Flat-field-corrected projection at an arbitrary index (CV_32FC1, no display scaling).
    cv::Mat get_projection_corr(int index);
    // Same result correctScan() caches for this index (spot filter + flat field + intensity
    // correction + phase retrieval), computed directly from scan/ - shared by correctScan() and
    // any caller that wants the fully corrected projection without going through the cache.
    cv::Mat get_projection_corrected_full(int index);

    cv::Mat  im_show;

private:
    cv::Mat ob_corr, di_corr;
    QString data_path;
    QRect roi_rect;
    int spot_kernel_size = 5;
    int binning = 1;
    std::vector<cv::Mat> correctedProjections_;

    QRect intensity_roi;
    bool intensity_roi_enabled = false;
    double intensity_reference = 0.0;
    // Mean pixel value of img within intensity_roi (clamped to img's bounds). Returns 0 if the
    // clamped region is empty.
    double roi_mean(const cv::Mat& img) const;

    // Reads `filename`, crops to fullResRoi, and if binning > 1, block-averages down to
    // (fullResRoi.width/binning, fullResRoi.height/binning). fullResRoi is always in the
    // original, unbinned image's coordinates.
    cv::Mat read_cropped_binned(const QString& filename, const cv::Rect& fullResRoi) const;

    // Area read from disk for every frame: the ROI itself, or - with a distortion correction - the
    // ROI grown by what the resampling needs (see DistortionCorrection::requiredSourceRect).
    cv::Rect workRoi() const;
    // Resamples a flat-fielded frame covering workRoi() (binned) onto the corrected pixel grid of
    // the ROI; returns `img` unchanged without a distortion correction.
    cv::Mat undistort(const cv::Mat& img);

    std::shared_ptr<const DistortionCorrection> distortion_;
    // Remap table for the current roi_rect/binning, built on first use and rebuilt if either changes.
    std::mutex distortion_mutex;
    bool distortion_roi_valid = false;
    QRect distortion_roi;
    int distortion_binning = 1;
    cv::Rect distortion_src;
    cv::Mat distortion_map;
    void prepareDistortion();

    bool use_gpu = false;
#if !defined(__APPLE__)
    std::unique_ptr<SpotFilterCudaBackend> spot_filter_gpu;
    int spot_filter_gpu_rows = 0, spot_filter_gpu_cols = 0;
    std::unique_ptr<PhaseRetrievalCudaBackend> phase_retrieval_gpu;
    int phase_retrieval_gpu_nx = 0, phase_retrieval_gpu_ny = 0;
#endif

    // Dispatches to Spot_filter::spot_correction (CPU) or spot_filter_gpu (GPU, rebuilt if img's
    // size doesn't match the cached backend) depending on use_gpu. Same call sites as before just
    // route through here instead of calling Spot_filter::spot_correction directly.
    void applySpotCorrection(cv::Mat& img, int kernel_size, int threshold);
    // Dispatches to the free function phase_retrieval() (CPU) or phase_retrieval_gpu (GPU, rebuilt
    // if nx/ny don't match the cached backend) depending on use_gpu.
    std::vector<float> applyPhaseRetrieval(const std::vector<float>& image, int nx, int ny, float alpha, float pix);
};

#endif // PROJ_CORRECTION_H
