#include "fbp_cuda_backend.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {

void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "FbpCudaBackend: " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

void cufftCheck(cufftResult err, const char* what)
{
    if (err != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << "FbpCudaBackend: " << what << " failed (cufftResult " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), #expr)
#define CUFFT_CHECK(expr) cufftCheck((expr), #expr)

// Frequency index in FFT sample order: 0, 1, ..., N/2, -(N/2-1), ..., -1.
// Mirrors fft_freq_index() in fbp_reconstructor.cpp - kept as a separate
// device-side copy since the two live in different translation units (this
// one compiled by nvcc, that one by the host compiler).
__device__ inline int fftFreqIndex(int k, int N)
{
    return (k <= N / 2) ? k : k - N;
}

__global__ void realToComplexPaddedKernel(const float* src, int n_detectors,
                                           cufftComplex* dst, int filterSize, int n_angles)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= n_detectors)
        return;
    size_t idx = static_cast<size_t>(row) * filterSize + col;
    dst[idx].x = src[static_cast<size_t>(row) * n_detectors + col];
    dst[idx].y = 0.0f;
}

__global__ void zeroComplexKernel(cufftComplex* buf, size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    buf[idx].x = 0.0f;
    buf[idx].y = 0.0f;
}

__global__ void applyShiftPhaseKernel(cufftComplex* buf, int filterSize, int n_angles, double shift)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= filterSize)
        return;
    int freq = fftFreqIndex(col, filterSize);
    double angle = -2.0 * M_PI * freq * shift / static_cast<double>(filterSize);
    float pr = static_cast<float>(cos(angle));
    float pi = static_cast<float>(sin(angle));
    size_t idx = static_cast<size_t>(row) * filterSize + col;
    cufftComplex v = buf[idx];
    cufftComplex out;
    out.x = v.x * pr - v.y * pi;
    out.y = v.x * pi + v.y * pr;
    buf[idx] = out;
}

__global__ void applyRampFilterKernel(cufftComplex* buf, const float* filter, int filterSize, int n_angles)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= filterSize)
        return;
    float f = filter[col];
    size_t idx = static_cast<size_t>(row) * filterSize + col;
    buf[idx].x *= f;
    buf[idx].y *= f;
}

__global__ void complexToRealUnpadKernel(const cufftComplex* src, int filterSize,
                                          float* dst, int n_detectors, int n_angles, float scale)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= n_detectors)
        return;
    dst[static_cast<size_t>(row) * n_detectors + col] =
        src[static_cast<size_t>(row) * filterSize + col].x * scale;
}

// One thread per output pixel. filteredSino rows are filterSize_ apart (only
// the first n_detectors columns of each hold real sinogram data - the rest is
// FFT zero-padding); only the real (.x) component is used, matching the CPU
// path which only ever reads the real plane after the inverse FFT.
__global__ void backprojectKernel(const cufftComplex* filteredSino, int filterSize,
                                   const float* cosA, const float* sinA, int n_angles,
                                   int n_detectors, double dtheta, double circMaskRatio,
                                   float* outSlice)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n_detectors || y >= n_detectors)
        return;

    const double cx = n_detectors / 2.0;
    const double cy = n_detectors / 2.0;
    const double detCenter = n_detectors / 2.0;
    const double xc = x - cx;
    const double yc = y - cy;

    const double radius = circMaskRatio * (n_detectors / 2.0);
    const double radius2 = radius * radius;
    if (xc * xc + yc * yc > radius2) {
        outSlice[static_cast<size_t>(y) * n_detectors + x] = 0.0f;
        return;
    }

    double sum = 0.0;
    for (int a = 0; a < n_angles; ++a) {
        double t = xc * cosA[a] + yc * sinA[a] + detCenter;
        int t0 = static_cast<int>(floor(t));
        if (t0 < 0 || t0 + 1 >= n_detectors)
            continue;
        double frac = t - t0;
        const cufftComplex* row = filteredSino + static_cast<size_t>(a) * filterSize;
        sum += row[t0].x * (1.0 - frac) + row[t0 + 1].x * frac;
    }
    outSlice[static_cast<size_t>(y) * n_detectors + x] = static_cast<float>(sum * dtheta);
}

dim3 grid2d(int w, int h, dim3 block)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

} // namespace

struct FbpCudaBackend::Impl
{
    int n_detectors_;
    int filterSize_;
    float* d_filter_ = nullptr;   // length filterSize_
    float* d_outSlice_ = nullptr; // length n_detectors_ * n_detectors_

    // reconstruction_worker.cpp's run() shares one FbpReconstructor (and so one FbpCudaBackend)
    // across a multi-threaded pool - every row reconstructed concurrently on the same instance.
    // The cuFFT plan and every scratch buffer below are mutable, shared, per-instance state (unlike
    // the old vDSP path, which only ever touched a read-only FFTSetup plus fresh per-call local
    // buffers), so concurrent calls must be serialized or they corrupt each other's buffers
    // mid-flight. The GPU is one device anyway, so serializing here costs little.
    std::mutex mutex_;

    // Batch (n_angles)-sized state, cached and rebuilt only when n_angles changes.
    int cachedBatch_ = 0;
    cufftHandle plan_{};
    cufftComplex* d_complexBuf_ = nullptr; // cachedBatch_ * filterSize_
    float* d_srcFlat_ = nullptr;           // cachedBatch_ * n_detectors_, tightly packed
    float* d_cosA_ = nullptr;              // cachedBatch_
    float* d_sinA_ = nullptr;              // cachedBatch_

    Impl(int n_detectors, int filterSize, const std::vector<float>& filter)
        : n_detectors_(n_detectors), filterSize_(filterSize)
    {
        CUDA_CHECK(cudaMalloc(&d_filter_, sizeof(float) * filterSize_));
        CUDA_CHECK(cudaMemcpy(d_filter_, filter.data(), sizeof(float) * filterSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_outSlice_, sizeof(float) * static_cast<size_t>(n_detectors_) * n_detectors_));
    }

    ~Impl()
    {
        freeBatch();
        cudaFree(d_filter_);
        cudaFree(d_outSlice_);
    }

    void freeBatch()
    {
        if (cachedBatch_ == 0)
            return;
        cufftDestroy(plan_);
        cudaFree(d_complexBuf_);
        cudaFree(d_srcFlat_);
        cudaFree(d_cosA_);
        cudaFree(d_sinA_);
        cachedBatch_ = 0;
    }

    void ensureBatch(int n_angles)
    {
        if (n_angles == cachedBatch_)
            return;
        freeBatch();
        CUFFT_CHECK(cufftPlan1d(&plan_, filterSize_, CUFFT_C2C, n_angles));
        CUDA_CHECK(cudaMalloc(&d_complexBuf_, sizeof(cufftComplex) * static_cast<size_t>(n_angles) * filterSize_));
        CUDA_CHECK(cudaMalloc(&d_srcFlat_, sizeof(float) * static_cast<size_t>(n_angles) * n_detectors_));
        CUDA_CHECK(cudaMalloc(&d_cosA_, sizeof(float) * n_angles));
        CUDA_CHECK(cudaMalloc(&d_sinA_, sizeof(float) * n_angles));
        cachedBatch_ = n_angles;
    }

    // Uploads `sinogram` (n_angles x n_detectors_, host row pitch rowStrideBytes) into
    // d_complexBuf_, zero-padded to filterSize_ columns per row.
    void uploadPadded(const float* sinogram, int n_angles, size_t rowStrideBytes)
    {
        CUDA_CHECK(cudaMemcpy2D(d_srcFlat_, n_detectors_ * sizeof(float),
                                 sinogram, rowStrideBytes,
                                 n_detectors_ * sizeof(float), n_angles,
                                 cudaMemcpyHostToDevice));

        size_t count = static_cast<size_t>(n_angles) * filterSize_;
        dim3 zblock(256);
        dim3 zgrid((count + zblock.x - 1) / zblock.x);
        zeroComplexKernel<<<zgrid, zblock>>>(d_complexBuf_, count);
        CUDA_CHECK(cudaGetLastError());

        dim3 block(32, 8);
        dim3 grid = grid2d(n_detectors_, n_angles, block);
        realToComplexPaddedKernel<<<grid, block>>>(d_srcFlat_, n_detectors_, d_complexBuf_, filterSize_, n_angles);
        CUDA_CHECK(cudaGetLastError());
    }
};

FbpCudaBackend::FbpCudaBackend(int n_detectors, int filter_size, const std::vector<float>& filter)
    : impl_(std::make_unique<Impl>(n_detectors, filter_size, filter))
{
}

FbpCudaBackend::~FbpCudaBackend() = default;

void FbpCudaBackend::shiftSinogram(float* sinogram, int n_angles, size_t rowStrideBytes, double shift)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);
    impl.ensureBatch(n_angles);
    impl.uploadPadded(sinogram, n_angles, rowStrideBytes);

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_complexBuf_, impl.d_complexBuf_, CUFFT_FORWARD));

    dim3 block(32, 8);
    dim3 grid = grid2d(impl.filterSize_, n_angles, block);
    applyShiftPhaseKernel<<<grid, block>>>(impl.d_complexBuf_, impl.filterSize_, n_angles, shift);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_complexBuf_, impl.d_complexBuf_, CUFFT_INVERSE));

    // cuFFT is unnormalized, same as the CPU/vDSP path was - scale by 1/filterSize_ while
    // unpacking back down to n_detectors_ columns.
    float scale = 1.0f / static_cast<float>(impl.filterSize_);
    dim3 ugrid = grid2d(impl.n_detectors_, n_angles, block);
    complexToRealUnpadKernel<<<ugrid, block>>>(impl.d_complexBuf_, impl.filterSize_, impl.d_srcFlat_,
                                                impl.n_detectors_, n_angles, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(sinogram, rowStrideBytes,
                             impl.d_srcFlat_, impl.n_detectors_ * sizeof(float),
                             impl.n_detectors_ * sizeof(float), n_angles,
                             cudaMemcpyDeviceToHost));
}

void FbpCudaBackend::reconstructSlice(const float* sinogram, int n_angles, size_t rowStrideBytes,
                                       const std::vector<double>& angles_rad, double circ_mask_ratio,
                                       float* outSlice)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);
    impl.ensureBatch(n_angles);
    impl.uploadPadded(sinogram, n_angles, rowStrideBytes);

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_complexBuf_, impl.d_complexBuf_, CUFFT_FORWARD));

    dim3 block(32, 8);
    dim3 fgrid = grid2d(impl.filterSize_, n_angles, block);
    applyRampFilterKernel<<<fgrid, block>>>(impl.d_complexBuf_, impl.d_filter_, impl.filterSize_, n_angles);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_complexBuf_, impl.d_complexBuf_, CUFFT_INVERSE));
    // Filtered sinogram's real part (already includes the 1/filterSize_ prescale baked into
    // d_filter_) stays in impl.d_complexBuf_ on the device - read directly by backprojectKernel,
    // no host round-trip.

    const double dtheta = (n_angles > 1) ? (angles_rad[1] - angles_rad[0]) : M_PI / n_angles;
    std::vector<float> cosA(n_angles), sinA(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        cosA[a] = static_cast<float>(std::cos(angles_rad[a]));
        sinA[a] = static_cast<float>(std::sin(angles_rad[a]));
    }
    CUDA_CHECK(cudaMemcpy(impl.d_cosA_, cosA.data(), sizeof(float) * n_angles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_sinA_, sinA.data(), sizeof(float) * n_angles, cudaMemcpyHostToDevice));

    dim3 bpBlock(16, 16);
    dim3 bpGrid = grid2d(impl.n_detectors_, impl.n_detectors_, bpBlock);
    backprojectKernel<<<bpGrid, bpBlock>>>(impl.d_complexBuf_, impl.filterSize_, impl.d_cosA_, impl.d_sinA_,
                                            n_angles, impl.n_detectors_, dtheta, circ_mask_ratio,
                                            impl.d_outSlice_);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(outSlice, impl.d_outSlice_,
                           sizeof(float) * static_cast<size_t>(impl.n_detectors_) * impl.n_detectors_,
                           cudaMemcpyDeviceToHost));
}
