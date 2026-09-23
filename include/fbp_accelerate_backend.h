#ifndef FBP_ACCELERATE_BACKEND_H
#define FBP_ACCELERATE_BACKEND_H
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cstddef>

// CPU-resident state for FbpReconstructor on macOS: a shared vDSP FFT setup
// (a read-only twiddle-factor table, safe for concurrent use) plus a copy of
// the precomputed windowed ramp filter, length filter_size, pre-scaled by
// 1/filter_size - same convention FbpCudaBackend expects, so
// fbp_reconstructor.cpp's filter-construction code is identical on both
// platforms and only the backend it hands the filter to differs.
// Implemented in fbp_accelerate_backend.cpp (Accelerate/vDSP), the macOS
// counterpart to FbpCudaBackend/fbp_reconstructor_cuda.cu on Linux.
//
// Unlike FbpCudaBackend, which serializes access to one shared GPU, this
// backend needs no locking: ReconstructionWorker's per-row thread pool calls
// into it concurrently, and every call after construction touches only its
// own stack-local buffers (the FFT setup itself is read-only).
class FbpAccelerateBackend
{
public:
    FbpAccelerateBackend(int n_detectors, int filter_size, const std::vector<float>& filter);
    ~FbpAccelerateBackend();

    FbpAccelerateBackend(const FbpAccelerateBackend&) = delete;
    FbpAccelerateBackend& operator=(const FbpAccelerateBackend&) = delete;

    // In-place Fourier sub-pixel shift of an (n_angles x n_detectors) sinogram.
    // rowStrideBytes is the host row pitch (may exceed n_detectors*sizeof(float)
    // for a non-contiguous cv::Mat).
    void shiftSinogram(float* sinogram, int n_angles, std::size_t rowStrideBytes, double shift);

    // Filters (windowed ramp) then backprojects an (n_angles x n_detectors)
    // sinogram into a contiguous n_detectors x n_detectors output slice.
    void reconstructSlice(const float* sinogram, int n_angles, std::size_t rowStrideBytes,
                           const std::vector<double>& angles_rad, double circ_mask_ratio,
                           float* outSlice);

private:
    int n_detectors_;
    int filter_size_;
    vDSP_Length log2_filter_size_;
    FFTSetup fft_setup_;
    std::vector<float> filter_;       // pre-scaled by 1/filter_size, see above

    void filter_row(const float* in, float* out) const;
};

#endif // FBP_ACCELERATE_BACKEND_H
