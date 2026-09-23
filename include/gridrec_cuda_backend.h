#ifndef GRIDREC_CUDA_BACKEND_H
#define GRIDREC_CUDA_BACKEND_H
#include <vector>
#include <memory>
#include <cstddef>

// GPU-resident state for GridrecReconstructor: device copies of the
// precomputed windowed ramp filter, detector-centering phase and
// deapodization curve GridrecReconstructor's constructor computes, plus a
// batched-1D cuFFT plan (row filtering, rebuilt only when the angle count
// changes) and a fixed-size 2D cuFFT plan (the gridding grid, built once -
// grid_size never changes for a given instance). Implemented in
// gridrec_reconstructor_cuda.cu (compiled by nvcc) so this header, included
// from ordinary C++ translation units, never pulls in CUDA/cuFFT headers or
// types - same pattern as FbpCudaBackend/fbp_cuda_backend.h.
class GridrecCudaBackend
{
public:
    // filter: precomputed windowed ramp, length grid_size (NOT pre-scaled by
    // 1/grid_size - unlike FbpCudaBackend's filter, gridrec's overall
    // amplitude correction happens once at the end of reconstructSlice, not
    // per filtered row). kb_width/kb_beta: Kaiser-Bessel gridding-kernel
    // parameters. deapod_1d: length grid_size, separable per-axis
    // deapodization curve. centerPhaseReal/Imag: length grid_size, the
    // detector-to-rotation-axis re-centering phase ramp.
    GridrecCudaBackend(int n_detectors, int grid_size, const std::vector<float>& filter,
                        double kb_width, double kb_beta, const std::vector<float>& deapod_1d,
                        const std::vector<float>& centerPhaseReal, const std::vector<float>& centerPhaseImag);
    ~GridrecCudaBackend();

    GridrecCudaBackend(const GridrecCudaBackend&) = delete;
    GridrecCudaBackend& operator=(const GridrecCudaBackend&) = delete;

    // In-place Fourier sub-pixel shift of an (n_angles x n_detectors) sinogram.
    // rowStrideBytes is the host row pitch (may exceed n_detectors*sizeof(float)
    // for a non-contiguous cv::Mat).
    void shiftSinogram(float* sinogram, int n_angles, std::size_t rowStrideBytes, double shift);

    // Filters (windowed ramp) then grids+FFTs an (n_angles x n_detectors)
    // sinogram into a contiguous n_detectors x n_detectors output slice.
    void reconstructSlice(const float* sinogram, int n_angles, std::size_t rowStrideBytes,
                           const std::vector<double>& angles_rad, double circ_mask_ratio,
                           float* outSlice);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // GRIDREC_CUDA_BACKEND_H
