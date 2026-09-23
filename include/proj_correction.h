#ifndef PROJ_CORRECTION_H
#define PROJ_CORRECTION_H
#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
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
    }
    // Spot (hot/dead pixel) filter window size, applied in load_op_di/get_projection_corr/run_scan.
    void setSpotKernelSize(int k) { spot_kernel_size = k; }
    // NxN block-average downsampling (1 = disabled), applied immediately after each image is
    // read and cropped to the ROI - before spot filtering, flat-fielding, or phase retrieval - so
    // every later step in this class runs on the smaller image too.
    void setBinning(int b) { binning = b; }
    // Runs spot filtering and phase retrieval on the GPU (SpotFilterCudaBackend/
    // PhaseRetrievalCudaBackend) instead of the CPU (Spot_filter/phase_retrieval) - see
    // applySpotCorrection/applyPhaseRetrieval. The GPU backends are (re)built lazily, on first use
    // or whenever the image size changes. No CUDA build on macOS (see the __APPLE__ guards below),
    // so there this is a harmless no-op and applySpotCorrection/applyPhaseRetrieval always take the
    // CPU path regardless of the flag - same pattern as ReconstructionWorker's polar ring GPU path.
    void setUseGpu(bool useGpu) { use_gpu = useGpu; }
    // Region of the corrected (cropped/binned) image containing only the primary beam, no sample -
    // used to correct for beam-intensity fluctuation over the course of a scan. Immediately reads
    // and corrects projection 0 to establish the reference mean; every projection returned by
    // get_projection_corr() and run_scan() afterwards (including index 0 itself) is rescaled so its
    // own mean in this region matches that reference. Coordinates are in the same pixel space as
    // im_show / get_projection_corr()'s output (post-crop, post-binning) - re-pick after changing
    // binning. Throws std::runtime_error if roi is empty or projection 0 can't be read.
    void setIntensityRoi(const QRect& roi);
    void load_op_di();
    void get_first_image_corr();
    // progressCallback, if given, is invoked as (doneCount, totalCount) after each projection is
    // written - intended for a caller on another thread to translate into a Qt progress signal.
    void run_scan(const std::function<void(int, int)>& progressCallback = nullptr);
    // Flat-field-corrected projection at an arbitrary index (CV_32FC1, no display scaling).
    cv::Mat get_projection_corr(int index);
    // Reads back projection `index` from <data_path>/corr/, already flat-field + phase-retrieval
    // corrected by run_scan() (CV_32FC1, no further correction applied).
    cv::Mat get_projection_from_corr(int index);

    cv::Mat  im_show;

private:
    cv::Mat ob_corr, di_corr;
    QString data_path;
    QRect roi_rect;
    int spot_kernel_size = 5;
    int binning = 1;

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
