#include "phase_retrieval_cuda.h"
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
        oss << "PhaseRetrievalCudaBackend: " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}
void cufftCheck(cufftResult err, const char* what)
{
    if (err != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << "PhaseRetrievalCudaBackend: " << what << " failed (cufftResult " << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr)
#define CUFFT_CHECK(expr) cufftCheck((expr), #expr)

// Real-valued Paganin filter, FFT-frequency ordered - identical formula to the CPU
// phase_retrieval()'s filter-building loop in proj_correction.cpp.
__global__ void buildFilterKernel(float* filter, int nx, int ny, float alpha, float pix)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column (0..ny-1)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row (0..nx-1)
    if (i >= nx || j >= ny)
        return;
    int kxi = (i <= nx / 2) ? i : i - nx;
    float kx = 2.0f * static_cast<float>(M_PI) * kxi / (nx * pix);
    float coskx = cosf(kx * pix);
    int kyi = (j <= ny / 2) ? j : j - ny;
    float ky = 2.0f * static_cast<float>(M_PI) * kyi / (ny * pix);
    float cosky = cosf(ky * pix);
    filter[static_cast<size_t>(i) * ny + j] = 1.0f - (2.0f * alpha / (pix * pix)) * (coskx + cosky - 2.0f);
}

__global__ void packRealKernel(const float* image, cufftComplex* data, size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    data[idx].x = image[idx];
    data[idx].y = 0.0f;
}

__global__ void divideByFilterKernel(cufftComplex* data, const float* filter, size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float f = filter[idx];
    data[idx].x /= f;
    data[idx].y /= f;
}

// Undoes cuFFT's unnormalized round-trip scaling (same as the CPU path), then magnitude + -log.
__global__ void magnitudeNegLogKernel(const cufftComplex* data, float* out, size_t count, float scale)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float re = data[idx].x, im = data[idx].y;
    float mag = sqrtf(re * re + im * im) * scale;
    out[idx] = -logf(mag);
}

} // namespace

struct PhaseRetrievalCudaBackend::Impl
{
    int nx_, ny_;
    cufftHandle plan_{};
    cufftComplex* d_data_ = nullptr;
    float* d_filter_ = nullptr;
    float* d_image_ = nullptr;
    float* d_out_ = nullptr;
    float cachedAlpha_ = 0.0f;
    float cachedPix_ = 0.0f;
    bool filterBuilt_ = false;
    std::mutex mutex_;

    Impl(int nx, int ny) : nx_(nx), ny_(ny)
    {
        size_t count = static_cast<size_t>(nx_) * ny_;
        CUFFT_CHECK(cufftPlan2d(&plan_, nx_, ny_, CUFFT_C2C));
        CUDA_CHECK(cudaMalloc(&d_data_, count * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_filter_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_image_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_, count * sizeof(float)));
    }

    ~Impl()
    {
        cufftDestroy(plan_);
        cudaFree(d_data_);
        cudaFree(d_filter_);
        cudaFree(d_image_);
        cudaFree(d_out_);
    }
};

PhaseRetrievalCudaBackend::PhaseRetrievalCudaBackend(int nx, int ny)
    : impl_(std::make_unique<Impl>(nx, ny))
{
}

PhaseRetrievalCudaBackend::~PhaseRetrievalCudaBackend() = default;

std::vector<float> PhaseRetrievalCudaBackend::retrieve(const std::vector<float>& image, float alpha, float pix)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);

    const size_t count = static_cast<size_t>(impl.nx_) * impl.ny_;
    if (image.size() != count)
        throw std::invalid_argument("PhaseRetrievalCudaBackend::retrieve: image size != nx*ny");

    if (!impl.filterBuilt_ || alpha != impl.cachedAlpha_ || pix != impl.cachedPix_) {
        dim3 block(32, 8);
        dim3 grid((impl.ny_ + block.x - 1) / block.x, (impl.nx_ + block.y - 1) / block.y);
        buildFilterKernel<<<grid, block>>>(impl.d_filter_, impl.nx_, impl.ny_, alpha, pix);
        CUDA_CHECK(cudaGetLastError());
        impl.cachedAlpha_ = alpha;
        impl.cachedPix_ = pix;
        impl.filterBuilt_ = true;
    }

    CUDA_CHECK(cudaMemcpy(impl.d_image_, image.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block1d(256);
    dim3 grid1d(static_cast<unsigned>((count + block1d.x - 1) / block1d.x));
    packRealKernel<<<grid1d, block1d>>>(impl.d_image_, impl.d_data_, count);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_data_, impl.d_data_, CUFFT_FORWARD));

    divideByFilterKernel<<<grid1d, block1d>>>(impl.d_data_, impl.d_filter_, count);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(impl.plan_, impl.d_data_, impl.d_data_, CUFFT_INVERSE));

    float scale = 1.0f / static_cast<float>(count);
    magnitudeNegLogKernel<<<grid1d, block1d>>>(impl.d_data_, impl.d_out_, count, scale);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), impl.d_out_, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
