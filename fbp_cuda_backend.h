#ifndef FBP_CUDA_BACKEND_H
#define FBP_CUDA_BACKEND_H
#include <vector>
#include <memory>
#include <cstddef>

// GPU-resident state for FbpReconstructor: a precomputed device copy of the
// windowed ramp filter, plus a cuFFT batched-1D plan and scratch device
// buffers that are cached and only reallocated when the angle count (batch
// size) changes between calls - it's constant within one reconstruction run.
// Implemented in fbp_reconstructor_cuda.cu (compiled by nvcc) so this header,
// included from ordinary C++ translation units, never pulls in CUDA/cuFFT
// headers or types.
class FbpCudaBackend
{
public:
    // filter: precomputed windowed ramp, length filter_size, pre-scaled by 1/filter_size
    // so the batched inverse FFT needs no separate scale step.
    FbpCudaBackend(int n_detectors, int filter_size, const std::vector<float>& filter);
    ~FbpCudaBackend();

    FbpCudaBackend(const FbpCudaBackend&) = delete;
    FbpCudaBackend& operator=(const FbpCudaBackend&) = delete;

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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // FBP_CUDA_BACKEND_H
