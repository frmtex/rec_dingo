#include "spot_filter_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {

void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "SpotFilterCudaBackend: " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr)

__global__ void u16ToFloatKernel(const uint16_t* src, float* dst, size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
        dst[idx] = static_cast<float>(src[idx]);
}

// One thread per pixel; BORDER_REPLICATE (clamp) at the edges, matching cv::erode/cv::dilate's
// border handling in the CPU path.
__global__ void minFilterKernel(const float* src, float* dst, int rows, int cols, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;
    float best = src[static_cast<size_t>(y) * cols + x];
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = min(max(y + dy, 0), rows - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = min(max(x + dx, 0), cols - 1);
            float v = src[static_cast<size_t>(yy) * cols + xx];
            best = min(best, v);
        }
    }
    dst[static_cast<size_t>(y) * cols + x] = best;
}

__global__ void maxFilterKernel(const float* src, float* dst, int rows, int cols, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;
    float best = src[static_cast<size_t>(y) * cols + x];
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = min(max(y + dy, 0), rows - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = min(max(x + dx, 0), cols - 1);
            float v = src[static_cast<size_t>(yy) * cols + xx];
            best = max(best, v);
        }
    }
    dst[static_cast<size_t>(y) * cols + x] = best;
}

__global__ void spotReplaceKernel(uint16_t* img, const float* reference, size_t count, float threshold)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float ref = reference[idx];
    float diff = fabsf(static_cast<float>(img[idx]) - ref);
    if (diff > threshold) {
        float clamped = fminf(fmaxf(ref, 0.0f), 65535.0f);
        img[idx] = static_cast<uint16_t>(clamped);
    }
}

dim3 grid2d(int w, int h, dim3 block)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

} // namespace

struct SpotFilterCudaBackend::Impl
{
    int rows_, cols_;
    uint16_t* d_img_ = nullptr;
    float* d_imgF_ = nullptr;
    float* d_eroded_ = nullptr;
    float* d_opened_ = nullptr;
    float* d_dilated_ = nullptr;
    float* d_openClosed_ = nullptr;
    std::mutex mutex_;

    Impl(int rows, int cols) : rows_(rows), cols_(cols)
    {
        size_t count = static_cast<size_t>(rows_) * cols_;
        CUDA_CHECK(cudaMalloc(&d_img_, count * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_imgF_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eroded_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_opened_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dilated_, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_openClosed_, count * sizeof(float)));
    }

    ~Impl()
    {
        cudaFree(d_img_);
        cudaFree(d_imgF_);
        cudaFree(d_eroded_);
        cudaFree(d_opened_);
        cudaFree(d_dilated_);
        cudaFree(d_openClosed_);
    }
};

SpotFilterCudaBackend::SpotFilterCudaBackend(int rows, int cols)
    : impl_(std::make_unique<Impl>(rows, cols))
{
}

SpotFilterCudaBackend::~SpotFilterCudaBackend() = default;

void SpotFilterCudaBackend::spot_correction(cv::Mat& img, int kernel_size, int threshold)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);

    CV_Assert(img.type() == CV_16UC1);
    CV_Assert(img.rows == impl.rows_ && img.cols == impl.cols_);

    const int radius = kernel_size / 2;
    const size_t count = static_cast<size_t>(impl.rows_) * impl.cols_;

    CUDA_CHECK(cudaMemcpy2D(impl.d_img_, impl.cols_ * sizeof(uint16_t), img.ptr<uint16_t>(0), img.step,
                             impl.cols_ * sizeof(uint16_t), impl.rows_, cudaMemcpyHostToDevice));

    dim3 block1d(256);
    dim3 grid1d(static_cast<unsigned>((count + block1d.x - 1) / block1d.x));
    u16ToFloatKernel<<<grid1d, block1d>>>(impl.d_img_, impl.d_imgF_, count);
    CUDA_CHECK(cudaGetLastError());

    dim3 block2d(32, 8);
    dim3 grid2 = grid2d(impl.cols_, impl.rows_, block2d);

    // Opening (erode then dilate) removes bright spikes; closing the opened image (dilate then
    // erode) removes dark pits - same order as Spot_filter::despeckle_reference.
    minFilterKernel<<<grid2, block2d>>>(impl.d_imgF_, impl.d_eroded_, impl.rows_, impl.cols_, radius);
    CUDA_CHECK(cudaGetLastError());
    maxFilterKernel<<<grid2, block2d>>>(impl.d_eroded_, impl.d_opened_, impl.rows_, impl.cols_, radius);
    CUDA_CHECK(cudaGetLastError());
    maxFilterKernel<<<grid2, block2d>>>(impl.d_opened_, impl.d_dilated_, impl.rows_, impl.cols_, radius);
    CUDA_CHECK(cudaGetLastError());
    minFilterKernel<<<grid2, block2d>>>(impl.d_dilated_, impl.d_openClosed_, impl.rows_, impl.cols_, radius);
    CUDA_CHECK(cudaGetLastError());

    spotReplaceKernel<<<grid1d, block1d>>>(impl.d_img_, impl.d_openClosed_, count,
                                            static_cast<float>(threshold));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(img.ptr<uint16_t>(0), img.step, impl.d_img_, impl.cols_ * sizeof(uint16_t),
                             impl.cols_ * sizeof(uint16_t), impl.rows_, cudaMemcpyDeviceToHost));
}
