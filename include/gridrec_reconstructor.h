#ifndef GRIDREC_RECONSTRUCTOR_H
#define GRIDREC_RECONSTRUCTOR_H
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "fbp_reconstructor.h"  // reuses FbpFilterType
#include "slice_reconstructor.h"

#if defined(__APPLE__)
class GridrecAccelerateBackend;
#else
class GridrecCudaBackend;
#endif

// Fourier-domain (Dowd/Marone "gridrec") reconstruction for parallel-beam
// tomography - see fbp_reconstructor.h for the shared cross-platform facade
// pattern this mirrors. Same ramp+window filtering as FbpReconstructor, but
// instead of O(n_angles * N^2) real-space backprojection, each filtered
// projection's 1D FFT is gridded (Kaiser-Bessel convolution) onto a 2D
// Cartesian frequency grid per the Fourier slice theorem, then a single 2D
// inverse FFT produces the slice: O(N^2 log N) total - much faster at high
// angle counts, at the cost of a small amount of gridding-kernel
// approximation error relative to FbpReconstructor's direct (exact
// interpolation) result.
//
// This class holds no platform-specific state itself: the ramp/window
// filter, the Kaiser-Bessel gridding-kernel parameters, its numerically
// derived deapodization curve, and the detector-to-rotation-axis centering
// phase are all plain host-side math (see gridrec_reconstructor.cpp),
// computed once here and handed to whichever backend this platform builds -
// on macOS, GridrecAccelerateBackend (Accelerate/vDSP); elsewhere,
// GridrecCudaBackend (cuFFT + custom gridding kernels).
class GridrecReconstructor : public SliceReconstructor
{
public:
    // n_detectors: number of columns in each sinogram row (detector width).
    explicit GridrecReconstructor(int n_detectors, FbpFilterType filterType = FbpFilterType::Hamming);
    ~GridrecReconstructor() override;

    GridrecReconstructor(const GridrecReconstructor&) = delete;
    GridrecReconstructor& operator=(const GridrecReconstructor&) = delete;

    // Same sub-pixel Fourier shift as FbpReconstructor (kept identical so
    // callers can swap reconstructors without changing COR-alignment code).
    void shift_sinogram(cv::Mat& sinogram, double shift) const override;

    // Filters (windowed ramp) then grids+FFTs a sinogram (n_angles x
    // n_detectors, CV_32FC1, already shifted onto the rotation axis) into a
    // reconstructed square slice of side n_detectors. angles_rad.size() must
    // equal sinogram.rows.
    cv::Mat reconstruct_slice(const cv::Mat& sinogram, const std::vector<double>& angles_rad,
                               double circ_mask_ratio = 0.995) const override;

private:
    int n_detectors_;
    int grid_size_;                   // oversampled 2D grid side, next pow2 >= 2*n_detectors_
#if defined(__APPLE__)
    std::unique_ptr<GridrecAccelerateBackend> backend_;
#else
    std::unique_ptr<GridrecCudaBackend> backend_;
#endif
};

#endif // GRIDREC_RECONSTRUCTOR_H
