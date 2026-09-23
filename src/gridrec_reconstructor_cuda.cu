#include "gridrec_cuda_backend.h"
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
        oss << "GridrecCudaBackend: " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

void cufftCheck(cufftResult err, const char* what)
{
    if (err != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << "GridrecCudaBackend: " << what << " failed (cufftResult " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), #expr)
#define CUFFT_CHECK(expr) cufftCheck((expr), #expr)

// Frequency index in FFT sample order: 0, 1, ..., N/2, -(N/2-1), ..., -1.
// Mirrors fft_freq_index() in gridrec_reconstructor.cpp - kept as a separate
// device-side copy since the two live in different translation units (this
// one compiled by nvcc, that one by the host compiler). Same duplication
// fbp_reconstructor_cuda.cu already uses for the identical reason.
__device__ inline int fftFreqIndex(int k, int N)
{
    return (k <= N / 2) ? k : k - N;
}

// Modified Bessel function of the first kind, order 0 - same Abramowitz &
// Stegun polynomial approximation as the host-side copies in
// gridrec_reconstructor.cpp / gridrec_accelerate_backend.cpp.
__device__ double besselI0(double x)
{
    double ax = fabs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
               + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    }
    double t = 3.75 / ax;
    return (exp(ax) / sqrt(ax))
           * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
              + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
              + t * (-0.01647633 + t * 0.00392377))))))));
}

// Kaiser-Bessel convolution weight at offset `d` (grid pixels) from a sample,
// kernel support `width` (total, i.e. nonzero for |d| <= width/2).
__device__ double kbWeight(double d, double width, double beta)
{
    double half = width / 2.0;
    if (fabs(d) >= half)
        return 0.0;
    double r = d / half;
    double arg = 1.0 - r * r;
    if (arg < 0.0)
        arg = 0.0;
    return besselI0(beta * sqrt(arg)) / besselI0(beta);
}

// --- The following four kernels are identical (by design) to
// fbp_reconstructor_cuda.cu's copies of the same name - duplicated here
// rather than shared across translation units, same as this codebase's
// existing precedent of each CUDA_SOURCES file being self-contained. ---

__global__ void realToComplexPaddedKernel(const float* src, int n_detectors,
                                           cufftComplex* dst, int gridSize, int n_angles)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= n_detectors)
        return;
    size_t idx = static_cast<size_t>(row) * gridSize + col;
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

__global__ void applyShiftPhaseKernel(cufftComplex* buf, int gridSize, int n_angles, double shift)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= gridSize)
        return;
    int freq = fftFreqIndex(col, gridSize);
    double angle = -2.0 * M_PI * freq * shift / static_cast<double>(gridSize);
    float pr = static_cast<float>(cos(angle));
    float pi = static_cast<float>(sin(angle));
    size_t idx = static_cast<size_t>(row) * gridSize + col;
    cufftComplex v = buf[idx];
    cufftComplex out;
    out.x = v.x * pr - v.y * pi;
    out.y = v.x * pi + v.y * pr;
    buf[idx] = out;
}

__global__ void complexToRealUnpadKernel(const cufftComplex* src, int gridSize,
                                          float* dst, int n_detectors, int n_angles, float scale)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= n_detectors)
        return;
    dst[static_cast<size_t>(row) * n_detectors + col] =
        src[static_cast<size_t>(row) * gridSize + col].x * scale;
}

// Multiplies each filtered row's spectrum by the windowed ramp filter, then
// rotates it by the detector-centering phase (see GridrecReconstructor's
// centerPhase comment) - both are per-frequency-bin (per-column) constants,
// shared across every row/angle, so one kernel does both in a single pass.
__global__ void applyRampAndCenterKernel(cufftComplex* buf, const float* filter,
                                          const float* centerPhaseReal, const float* centerPhaseImag,
                                          int gridSize, int n_angles)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_angles || col >= gridSize)
        return;
    size_t idx = static_cast<size_t>(row) * gridSize + col;
    cufftComplex v = buf[idx];
    float f = filter[col];
    float a = v.x * f, b = v.y * f;
    float cp = centerPhaseReal[col], sp = centerPhaseImag[col];
    cufftComplex out;
    out.x = a * cp - b * sp;
    out.y = a * sp + b * cp;
    buf[idx] = out;
}

// One thread per (angle, frequency-bin) sample. Fourier slice theorem:
// FT_1D(filtered projection)(freq_k) equals the 2D spectrum sampled along
// the line through the origin at this angle - scatters each sample onto
// nearby grid cells with a 2D separable Kaiser-Bessel weight (gridding
// convolution). Multiple samples can land on the same grid cell (adjacent
// angles' kernel footprints overlap), so accumulation needs atomicAdd -
// `grid` must already be zeroed (via zeroComplexKernel) before this runs.
__global__ void gridScatterKernel(const cufftComplex* rowSpectrum, const float* cosA, const float* sinA,
                                   int n_angles, int gridSize, double kbWidth, double kbBeta, int kr,
                                   cufftComplex* grid)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // frequency bin k
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // angle index a
    if (row >= n_angles || col >= gridSize)
        return;

    int freqK = fftFreqIndex(col, gridSize);
    double c = static_cast<double>(cosA[row]);
    double s = static_cast<double>(sinA[row]);
    double u = freqK * c;
    double v = freqK * s;
    int u0 = static_cast<int>(floor(u));
    int v0 = static_cast<int>(floor(v));

    cufftComplex val = rowSpectrum[static_cast<size_t>(row) * gridSize + col];

    for (int dv = -kr + 1; dv <= kr; ++dv) {
        double wv = kbWeight(v0 + dv - v, kbWidth, kbBeta);
        if (wv == 0.0)
            continue;
        int iv = ((v0 + dv) % gridSize + gridSize) % gridSize;
        for (int du = -kr + 1; du <= kr; ++du) {
            double wu = kbWeight(u0 + du - u, kbWidth, kbBeta);
            if (wu == 0.0)
                continue;
            int iu = ((u0 + du) % gridSize + gridSize) % gridSize;
            double w = wu * wv;
            size_t idx = static_cast<size_t>(iv) * gridSize + iu;
            atomicAdd(&grid[idx].x, static_cast<float>(w * val.x));
            atomicAdd(&grid[idx].y, static_cast<float>(w * val.y));
        }
    }
}

// One thread per output pixel. The 2D inverse FFT above was fed a grid in
// standard FFT order, so its output is also in that order: object center
// (x=0) sits at array index 0 and wraps to the high end for negative
// coordinates - fftshift it back to a natural layout (center at gridSize/2)
// while reading, same as GridrecAccelerateBackend's host-side loop.
__global__ void deapodCropMaskKernel(const cufftComplex* grid, int gridSize, const float* deapod,
                                      int n_detectors, double circMaskRatio, float scale, float* outSlice)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n_detectors || y >= n_detectors)
        return;

    size_t outIdx = static_cast<size_t>(y) * n_detectors + x;

    const double cx = n_detectors / 2.0;
    const double cy = n_detectors / 2.0;
    const double xc = x - cx;
    const double yc = y - cy;
    const double radius = circMaskRatio * (n_detectors / 2.0);
    const double radius2 = radius * radius;
    if (xc * xc + yc * yc > radius2) {
        outSlice[outIdx] = 0.0f;
        return;
    }

    const int half = gridSize / 2;
    const int off = (gridSize - n_detectors) / 2;
    int gy = (off + y + half) % gridSize;
    int gx = (off + x + half) % gridSize;
    float val = grid[static_cast<size_t>(gy) * gridSize + gx].x;
    outSlice[outIdx] = val * deapod[off + y] * deapod[off + x] * scale;
}

dim3 grid2d(int w, int h, dim3 block)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

} // namespace

struct GridrecCudaBackend::Impl
{
    int n_detectors_;
    int gridSize_;
    double kbWidth_;
    double kbBeta_;
    int kr_;

    float* d_filter_ = nullptr;          // length gridSize_
    float* d_deapod_ = nullptr;          // length gridSize_
    float* d_centerPhaseReal_ = nullptr; // length gridSize_
    float* d_centerPhaseImag_ = nullptr; // length gridSize_
    cufftComplex* d_gridComplex_ = nullptr; // gridSize_ * gridSize_, the 2D gridding accumulator
    float* d_outSlice_ = nullptr;        // n_detectors_ * n_detectors_
    cufftHandle plan2d_{};               // fixed gridSize_ x gridSize_ C2C, built once (batch=1)

    // reconstruction_worker.cpp's run() shares one GridrecReconstructor (and so one
    // GridrecCudaBackend) across a multi-threaded row pool - every row reconstructed concurrently
    // on the same instance. Every buffer/plan below is mutable, shared, per-instance state, so
    // concurrent calls must be serialized or they corrupt each other's buffers mid-flight - same
    // reasoning as FbpCudaBackend's mutex_.
    std::mutex mutex_;

    // Batch (n_angles)-sized state, cached and rebuilt only when n_angles changes.
    int cachedBatch_ = 0;
    cufftHandle plan1d_{};
    cufftComplex* d_rowComplexBuf_ = nullptr; // cachedBatch_ * gridSize_
    float* d_srcFlat_ = nullptr;              // cachedBatch_ * n_detectors_, tightly packed
    float* d_cosA_ = nullptr;                 // cachedBatch_
    float* d_sinA_ = nullptr;                 // cachedBatch_

    Impl(int n_detectors, int gridSize, const std::vector<float>& filter, double kbWidth, double kbBeta,
         const std::vector<float>& deapod_1d, const std::vector<float>& centerPhaseReal,
         const std::vector<float>& centerPhaseImag)
        : n_detectors_(n_detectors), gridSize_(gridSize), kbWidth_(kbWidth), kbBeta_(kbBeta)
    {
        kr_ = static_cast<int>(std::ceil(kbWidth_ / 2.0));

        CUDA_CHECK(cudaMalloc(&d_filter_, sizeof(float) * gridSize_));
        CUDA_CHECK(cudaMemcpy(d_filter_, filter.data(), sizeof(float) * gridSize_, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_deapod_, sizeof(float) * gridSize_));
        CUDA_CHECK(cudaMemcpy(d_deapod_, deapod_1d.data(), sizeof(float) * gridSize_, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_centerPhaseReal_, sizeof(float) * gridSize_));
        CUDA_CHECK(cudaMemcpy(d_centerPhaseReal_, centerPhaseReal.data(), sizeof(float) * gridSize_,
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_centerPhaseImag_, sizeof(float) * gridSize_));
        CUDA_CHECK(cudaMemcpy(d_centerPhaseImag_, centerPhaseImag.data(), sizeof(float) * gridSize_,
                               cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_gridComplex_, sizeof(cufftComplex) * static_cast<size_t>(gridSize_) * gridSize_));
        CUDA_CHECK(cudaMalloc(&d_outSlice_, sizeof(float) * static_cast<size_t>(n_detectors_) * n_detectors_));

        CUFFT_CHECK(cufftPlan2d(&plan2d_, gridSize_, gridSize_, CUFFT_C2C));
    }

    ~Impl()
    {
        freeBatch();
        cufftDestroy(plan2d_);
        cudaFree(d_filter_);
        cudaFree(d_deapod_);
        cudaFree(d_centerPhaseReal_);
        cudaFree(d_centerPhaseImag_);
        cudaFree(d_gridComplex_);
        cudaFree(d_outSlice_);
    }

    void freeBatch()
    {
        if (cachedBatch_ == 0)
            return;
        cufftDestroy(plan1d_);
        cudaFree(d_rowComplexBuf_);
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
        CUFFT_CHECK(cufftPlan1d(&plan1d_, gridSize_, CUFFT_C2C, n_angles));
        CUDA_CHECK(cudaMalloc(&d_rowComplexBuf_, sizeof(cufftComplex) * static_cast<size_t>(n_angles) * gridSize_));
        CUDA_CHECK(cudaMalloc(&d_srcFlat_, sizeof(float) * static_cast<size_t>(n_angles) * n_detectors_));
        CUDA_CHECK(cudaMalloc(&d_cosA_, sizeof(float) * n_angles));
        CUDA_CHECK(cudaMalloc(&d_sinA_, sizeof(float) * n_angles));
        cachedBatch_ = n_angles;
    }

    // Uploads `sinogram` (n_angles x n_detectors_, host row pitch rowStrideBytes) into
    // d_rowComplexBuf_, zero-padded to gridSize_ columns per row.
    void uploadPadded(const float* sinogram, int n_angles, size_t rowStrideBytes)
    {
        CUDA_CHECK(cudaMemcpy2D(d_srcFlat_, n_detectors_ * sizeof(float),
                                 sinogram, rowStrideBytes,
                                 n_detectors_ * sizeof(float), n_angles,
                                 cudaMemcpyHostToDevice));

        size_t count = static_cast<size_t>(n_angles) * gridSize_;
        dim3 zblock(256);
        dim3 zgrid((count + zblock.x - 1) / zblock.x);
        zeroComplexKernel<<<zgrid, zblock>>>(d_rowComplexBuf_, count);
        CUDA_CHECK(cudaGetLastError());

        dim3 block(32, 8);
        dim3 grid = grid2d(n_detectors_, n_angles, block);
        realToComplexPaddedKernel<<<grid, block>>>(d_srcFlat_, n_detectors_, d_rowComplexBuf_, gridSize_, n_angles);
        CUDA_CHECK(cudaGetLastError());
    }
};

GridrecCudaBackend::GridrecCudaBackend(int n_detectors, int grid_size, const std::vector<float>& filter,
                                        double kb_width, double kb_beta, const std::vector<float>& deapod_1d,
                                        const std::vector<float>& centerPhaseReal,
                                        const std::vector<float>& centerPhaseImag)
    : impl_(std::make_unique<Impl>(n_detectors, grid_size, filter, kb_width, kb_beta, deapod_1d,
                                    centerPhaseReal, centerPhaseImag))
{
}

GridrecCudaBackend::~GridrecCudaBackend() = default;

void GridrecCudaBackend::shiftSinogram(float* sinogram, int n_angles, size_t rowStrideBytes, double shift)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);
    impl.ensureBatch(n_angles);
    impl.uploadPadded(sinogram, n_angles, rowStrideBytes);

    CUFFT_CHECK(cufftExecC2C(impl.plan1d_, impl.d_rowComplexBuf_, impl.d_rowComplexBuf_, CUFFT_FORWARD));

    dim3 block(32, 8);
    dim3 grid = grid2d(impl.gridSize_, n_angles, block);
    applyShiftPhaseKernel<<<grid, block>>>(impl.d_rowComplexBuf_, impl.gridSize_, n_angles, shift);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(impl.plan1d_, impl.d_rowComplexBuf_, impl.d_rowComplexBuf_, CUFFT_INVERSE));

    // cuFFT is unnormalized, same as the CPU/vDSP path - scale by 1/gridSize_ while unpacking
    // back down to n_detectors_ columns.
    float scale = 1.0f / static_cast<float>(impl.gridSize_);
    dim3 ugrid = grid2d(impl.n_detectors_, n_angles, block);
    complexToRealUnpadKernel<<<ugrid, block>>>(impl.d_rowComplexBuf_, impl.gridSize_, impl.d_srcFlat_,
                                                impl.n_detectors_, n_angles, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(sinogram, rowStrideBytes,
                             impl.d_srcFlat_, impl.n_detectors_ * sizeof(float),
                             impl.n_detectors_ * sizeof(float), n_angles,
                             cudaMemcpyDeviceToHost));
}

void GridrecCudaBackend::reconstructSlice(const float* sinogram, int n_angles, size_t rowStrideBytes,
                                           const std::vector<double>& angles_rad, double circ_mask_ratio,
                                           float* outSlice)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);
    impl.ensureBatch(n_angles);
    impl.uploadPadded(sinogram, n_angles, rowStrideBytes);

    // 1. Filter every projection row (windowed ramp) and re-center it onto the rotation axis.
    CUFFT_CHECK(cufftExecC2C(impl.plan1d_, impl.d_rowComplexBuf_, impl.d_rowComplexBuf_, CUFFT_FORWARD));

    dim3 block(32, 8);
    dim3 fgrid = grid2d(impl.gridSize_, n_angles, block);
    applyRampAndCenterKernel<<<fgrid, block>>>(impl.d_rowComplexBuf_, impl.d_filter_, impl.d_centerPhaseReal_,
                                                impl.d_centerPhaseImag_, impl.gridSize_, n_angles);
    CUDA_CHECK(cudaGetLastError());

    // 2. Grid: scatter each filtered row's spectrum onto the 2D frequency grid (Fourier slice
    // theorem), accumulating via atomicAdd since adjacent angles' kernel footprints overlap.
    const double dtheta = (n_angles > 1) ? (angles_rad[1] - angles_rad[0]) : M_PI / n_angles;
    std::vector<float> cosA(n_angles), sinA(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        cosA[a] = static_cast<float>(std::cos(angles_rad[a]));
        sinA[a] = static_cast<float>(std::sin(angles_rad[a]));
    }
    CUDA_CHECK(cudaMemcpy(impl.d_cosA_, cosA.data(), sizeof(float) * n_angles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_sinA_, sinA.data(), sizeof(float) * n_angles, cudaMemcpyHostToDevice));

    size_t gridCount = static_cast<size_t>(impl.gridSize_) * impl.gridSize_;
    dim3 zblock(256);
    dim3 zgrid((gridCount + zblock.x - 1) / zblock.x);
    zeroComplexKernel<<<zgrid, zblock>>>(impl.d_gridComplex_, gridCount);
    CUDA_CHECK(cudaGetLastError());

    dim3 sgrid = grid2d(impl.gridSize_, n_angles, block);
    gridScatterKernel<<<sgrid, block>>>(impl.d_rowComplexBuf_, impl.d_cosA_, impl.d_sinA_, n_angles,
                                         impl.gridSize_, impl.kbWidth_, impl.kbBeta_, impl.kr_,
                                         impl.d_gridComplex_);
    CUDA_CHECK(cudaGetLastError());

    // 3. Single 2D inverse FFT of the gridded spectrum (fixed-size plan, built once in the
    // constructor - gridSize_ never changes for a given instance).
    CUFFT_CHECK(cufftExecC2C(impl.plan2d_, impl.d_gridComplex_, impl.d_gridComplex_, CUFFT_INVERSE));

    // 4. Deapodize, fftshift-crop to n_detectors_ x n_detectors_, circular mask. Same overall
    // scale-factor reasoning as GridrecAccelerateBackend: the 2D IFFT's N^2 unnormalized gain is
    // only half-cancelled by deapod_1d's own unnormalized-DFT construction, leaving one factor
    // of gridSize_ to correct here (see gridrec_reconstructor.cpp's deapod_1d comment).
    float scale = static_cast<float>(dtheta / static_cast<double>(impl.gridSize_));
    dim3 dGrid = grid2d(impl.n_detectors_, impl.n_detectors_, block);
    deapodCropMaskKernel<<<dGrid, block>>>(impl.d_gridComplex_, impl.gridSize_, impl.d_deapod_,
                                            impl.n_detectors_, circ_mask_ratio, scale, impl.d_outSlice_);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(outSlice, impl.d_outSlice_,
                           sizeof(float) * static_cast<size_t>(impl.n_detectors_) * impl.n_detectors_,
                           cudaMemcpyDeviceToHost));
}
