#ifndef SPOT_FILTER_CUDA_H
#define SPOT_FILTER_CUDA_H
#include <opencv2/opencv.hpp>
#include <memory>

// GPU-backed implementation of Spot_filter::spot_correction (see spot_filter.h for the algorithm
// description) - same morphological open-close despeckling, same BORDER_REPLICATE boundary, same
// threshold-and-replace logic, just as CUDA kernels (a min-filter and a max-filter, one thread per
// pixel - no sorting needed, unlike the ring filter's median) instead of OpenCV's CPU erode/dilate.
//
// One instance is meant to be constructed once per scan/session (sized by the cropped/binned
// projection dimensions, which are fixed for the whole scan) and reused across every projection
// and ob/di frame - mirrors FbpCudaBackend/PolarRingCudaBackend's design, including the mutex: this
// class's device buffers are shared, mutable, per-instance state, so concurrent calls (should a
// future caller ever make them) are serialized rather than left to race.
class SpotFilterCudaBackend
{
public:
    // rows, cols: the cropped/binned projection size for this scan.
    SpotFilterCudaBackend(int rows, int cols);
    ~SpotFilterCudaBackend();

    SpotFilterCudaBackend(const SpotFilterCudaBackend&) = delete;
    SpotFilterCudaBackend& operator=(const SpotFilterCudaBackend&) = delete;

    // Same signature/semantics as Spot_filter::spot_correction: img must be CV_16UC1 (may be a
    // non-contiguous ROI view - handled via a strided upload/download) and is modified in place.
    void spot_correction(cv::Mat& img, int kernel_size, int threshold);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // SPOT_FILTER_CUDA_H
