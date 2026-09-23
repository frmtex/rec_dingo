#ifndef GRIDREC_ACCELERATE_BACKEND_H
#define GRIDREC_ACCELERATE_BACKEND_H
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cstddef>

// CPU-resident state for GridrecReconstructor on macOS: a shared vDSP FFT
// setup (a read-only twiddle-factor table, safe for concurrent use, same as
// FbpAccelerateBackend's) plus copies of the precomputed windowed ramp
// filter, Kaiser-Bessel gridding-kernel parameters, deapodization curve, and
// detector-centering phase GridrecReconstructor's constructor computes -
// same convention GridrecCudaBackend expects, so gridrec_reconstructor.cpp's
// filter/kernel-construction code is identical on both platforms and only
// the backend it hands the results to differs.
// Implemented in gridrec_accelerate_backend.cpp (Accelerate/vDSP), the macOS
// counterpart to GridrecCudaBackend/gridrec_reconstructor_cuda.cu on Linux.
//
// Like FbpAccelerateBackend, this backend needs no locking: every call after
// construction touches only its own stack/heap-local buffers (the FFT setup
// itself is read-only), so ReconstructionWorker's per-row thread pool can
// call into it concurrently.
class GridrecAccelerateBackend
{
public:
    GridrecAccelerateBackend(int n_detectors, int grid_size, const std::vector<float>& filter,
                              double kb_width, double kb_beta, const std::vector<float>& deapod_1d,
                              const std::vector<float>& centerPhaseReal, const std::vector<float>& centerPhaseImag);
    ~GridrecAccelerateBackend();

    GridrecAccelerateBackend(const GridrecAccelerateBackend&) = delete;
    GridrecAccelerateBackend& operator=(const GridrecAccelerateBackend&) = delete;

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
    int n_detectors_;
    int grid_size_;
    vDSP_Length log2_grid_;
    FFTSetup fft_setup_;
    std::vector<float> filter_;
    double kb_width_;
    double kb_beta_;
    std::vector<float> deapod_1d_;
    std::vector<float> centerPhaseReal_, centerPhaseImag_;

    // FFTs+filters+re-centers one projection row, leaving it in the
    // frequency domain (unlike FbpAccelerateBackend::filter_row, which
    // inverse-transforms back to space domain for real-space backprojection).
    void filterRowFreq(const float* in, float* outReal, float* outImag) const;
};

#endif // GRIDREC_ACCELERATE_BACKEND_H
