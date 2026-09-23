#ifndef SLICE_RECONSTRUCTOR_H
#define SLICE_RECONSTRUCTOR_H
#include <opencv2/opencv.hpp>
#include <vector>

// Common interface implemented by FbpReconstructor and GridrecReconstructor,
// so callers (ReconstructionWorker) can pick an algorithm at runtime through
// one code path instead of duplicating each call site per algorithm.
class SliceReconstructor
{
public:
    virtual ~SliceReconstructor() = default;

    virtual void shift_sinogram(cv::Mat& sinogram, double shift) const = 0;

    virtual cv::Mat reconstruct_slice(const cv::Mat& sinogram, const std::vector<double>& angles_rad,
                                       double circ_mask_ratio) const = 0;
};

#endif // SLICE_RECONSTRUCTOR_H
