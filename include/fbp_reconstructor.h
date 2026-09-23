#ifndef FBP_RECONSTRUCTOR_H
#define FBP_RECONSTRUCTOR_H
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "slice_reconstructor.h"

// Standard apodization windows applied to the ideal ramp filter (Kak & Slaney
// convention, as a function of normalized frequency f in [-1, 1], f=0 at DC,
// f=+-1 at Nyquist). RamLak is the unwindowed ramp itself (max resolution,
// max noise); each window after it trades some resolution for noise
// suppression, roughly in increasing order of how aggressively it tapers
// high frequencies: SheppLogan, Cosine, Hamming, Hann (steepest taper, exact
// zero at Nyquist).
enum class FbpFilterType
{
    RamLak,
    SheppLogan,
    Cosine,
    Hamming,
    Hann
};

#if defined(__APPLE__)
class FbpAccelerateBackend;
#else
class FbpCudaBackend;
#endif

// Filtered backprojection (selectable apodization window on the ramp filter)
// for parallel-beam tomography, plus a Fourier-shift helper to move a
// sinogram onto the rotation axis before reconstruction. The ramp filter and
// backprojection run on whichever backend this platform builds: on Linux,
// FbpCudaBackend (cuFFT + CUDA kernels); on macOS, FbpAccelerateBackend
// (Accelerate/vDSP). Either way, the backend's cached FFT plan and scratch
// buffers are created once and reused across every sinogram/slice passed
// through this instance - this class itself holds no platform-specific state,
// just the shared filter-construction math and a pointer to whichever
// backend fbp_reconstructor.cpp was compiled against.
class FbpReconstructor : public SliceReconstructor
{
public:
    // n_detectors: number of columns in each sinogram row (detector width).
    explicit FbpReconstructor(int n_detectors, FbpFilterType filterType = FbpFilterType::Hamming);
    ~FbpReconstructor() override;

    FbpReconstructor(const FbpReconstructor&) = delete;
    FbpReconstructor& operator=(const FbpReconstructor&) = delete;

    // Shifts a sinogram (n_angles x n_detectors, CV_32FC1) horizontally by
    // `shift` pixels (sub-pixel, via the Fourier shift theorem), in place.
    void shift_sinogram(cv::Mat& sinogram, double shift) const override;

    // Filters (Hamming-windowed ramp) then backprojects a sinogram
    // (n_angles x n_detectors, CV_32FC1, already shifted onto the rotation
    // axis) into a reconstructed square slice of side n_detectors.
    // angles_rad.size() must equal sinogram.rows.
    cv::Mat reconstruct_slice(const cv::Mat& sinogram, const std::vector<double>& angles_rad,
                               double circ_mask_ratio = 0.995) const override;

private:
    int n_detectors_;
    int filter_size_;                 // next power of two >= 2*n_detectors_
#if defined(__APPLE__)
    std::unique_ptr<FbpAccelerateBackend> backend_;
#else
    std::unique_ptr<FbpCudaBackend> backend_;
#endif
};

#endif // FBP_RECONSTRUCTOR_H
