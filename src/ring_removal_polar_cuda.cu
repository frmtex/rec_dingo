#include "ring_removal_polar_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {

void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "PolarRingCudaBackend: " << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr)

// Max radial-median window this kernel supports (2*ringWidth+1 at the outer band, ringWidth up to
// the UI's spinBox_ringWidth maximum of 500 -> window up to 2*1001+1 = 2003), sized with margin.
constexpr int kMaxRadialWindow = 2048;

__device__ inline int bandedKernelRadius(int col, int polW, int design)
{
    int b1 = polW / 3;
    int b2 = (2 * polW) / 3;
    if (col < b1) return design / 3;
    if (col < b2) return (2 * design) / 3;
    return design;
}

__global__ void polarTransformKernel(const float* cartesianIn, int rows, int cols,
                                      float* polar, int polH, int polW,
                                      double centerX, double centerY, double threshMax, double threshMin)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= polH || r >= polW)
        return;
    double theta = a * 2.0 * M_PI / polH + M_PI / polH;
    double c = cos(theta), s = sin(theta);
    int xi = static_cast<int>(lround(r * c + centerX));
    int yi = static_cast<int>(lround(r * s + centerY));
    xi = max(0, min(cols - 1, xi));
    yi = max(0, min(rows - 1, yi));
    double val = cartesianIn[yi * cols + xi];
    val = max(threshMin, min(threshMax, val));
    polar[a * polW + r] = static_cast<float>(val);
}

// One thread per polar pixel. Window values are gathered into a fixed-size local array, then the
// median is found by counting each candidate's rank (ties broken by array index) - O(k^2) but
// fully parallel across pixels and simple to verify, unlike an in-place partial sort in device code.
__global__ void radialMedianKernel(const float* polar, int polH, int polW, int design, float* median)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= polH || c >= polW)
        return;

    int kr = bandedKernelRadius(c, polW, design);
    int windowSize = 2 * kr + 1;
    if (windowSize > kMaxRadialWindow)
        windowSize = kMaxRadialWindow; // clamps only for pathological ringWidth settings

    float window[kMaxRadialWindow];
    const float* rowPtr = polar + static_cast<size_t>(a) * polW;
    const float* oppositeRowPtr = polar + static_cast<size_t>((a + polH / 2) % polH) * polW;
    for (int k = -kr, w = 0; k <= kr && w < windowSize; ++k, ++w) {
        int idx = c + k;
        if (idx < 0)
            window[w] = oppositeRowPtr[min(-idx, polW - 1)];
        else if (idx >= polW)
            window[w] = rowPtr[polW - 1];
        else
            window[w] = rowPtr[idx];
    }

    int mid = windowSize / 2;
    float medianVal = window[0];
    for (int i = 0; i < windowSize; ++i) {
        int rank = 0;
        for (int j = 0; j < windowSize; ++j) {
            if (window[j] < window[i] || (window[j] == window[i] && j < i))
                ++rank;
        }
        if (rank == mid) {
            medianVal = window[i];
            break;
        }
    }
    median[a * polW + c] = medianVal;
}

__global__ void diffThresholdKernel(const float* polar, const float* median, size_t count,
                                     double thresh, float* diff)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    float v = polar[idx] - median[idx];
    diff[idx] = (fabsf(v) > thresh) ? 0.0f : v;
}

__device__ inline int reflectIndex(int idx, int polH)
{
    while (idx < 0 || idx >= polH) {
        if (idx < 0) idx = -idx - 1;
        if (idx >= polH) idx = 2 * polH - 1 - idx;
    }
    return idx;
}

__global__ void azimuthalMeanKernel(const float* diff, int polH, int polW, int design,
                                     bool wrapBoundary, float* outEstimate)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= polH || c >= polW)
        return;

    float self = diff[a * polW + c];
    if (self == 0.0f) {
        outEstimate[a * polW + c] = 0.0f;
        return;
    }
    int kr = bandedKernelRadius(c, polW, design);
    int windowSize = 2 * kr + 1;
    double sum = 0.0;
    for (int k = -kr; k <= kr; ++k) {
        int idx = a + k;
        idx = wrapBoundary ? ((idx % polH) + polH) % polH : reflectIndex(idx, polH);
        sum += diff[idx * polW + c];
    }
    outEstimate[a * polW + c] = static_cast<float>(sum / windowSize);
}

__global__ void inversePolarTransformKernel(const float* ringEstimatePolar, int polH, int polW,
                                             int rows, int cols, double centerX, double centerY,
                                             float* outCartesian)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;
    double dy = y - centerY;
    double dx = x - centerX;
    double theta = atan2(dy, dx) - M_PI / polH;
    if (theta < 0.0)
        theta += 2.0 * M_PI;
    int polRow = static_cast<int>(lround(theta * polH / (2.0 * M_PI)));
    if (polRow >= polH) polRow -= polH;
    if (polRow < 0) polRow += polH;
    int polCol = static_cast<int>(lround(sqrt(dx * dx + dy * dy)));
    float val = 0.0f;
    if (polRow >= 0 && polRow < polH && polCol >= 0 && polCol < polW)
        val = ringEstimatePolar[polRow * polW + polCol];
    outCartesian[y * cols + x] = val;
}

__global__ void subtractKernel(float* cartesianInOut, const float* ringEstimateCartesian, size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    cartesianInOut[idx] -= ringEstimateCartesian[idx];
}

dim3 grid2d(int w, int h, dim3 block)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

} // namespace

struct PolarRingCudaBackend::Impl
{
    int rows_, cols_;
    int polH_, polW_;
    double centerX_, centerY_;

    float* d_cartesianIn_ = nullptr;         // rows_*cols_ - also reused as the final output buffer
    float* d_polar_ = nullptr;               // polH_*polW_
    float* d_median_ = nullptr;              // polH_*polW_
    float* d_diff_ = nullptr;                // polH_*polW_
    float* d_ringEstimatePolar_ = nullptr;   // polH_*polW_
    float* d_ringEstimateCartesian_ = nullptr; // rows_*cols_

    std::mutex mutex_;

    Impl(int rows, int cols, double maskRadiusRatio) : rows_(rows), cols_(cols)
    {
        centerX_ = (cols_ - 1) / 2.0;
        centerY_ = (rows_ - 1) / 2.0;
        int geometricMaxRadius = static_cast<int>(std::floor(
            std::min({ centerX_, (cols_ - 1) - centerX_, centerY_, (rows_ - 1) - centerY_ })));
        int maskRadius = static_cast<int>(std::floor(maskRadiusRatio * std::min(rows_, cols_) / 2.0));
        int maxRadius = std::min(geometricMaxRadius, maskRadius);
        polW_ = std::max(1, maxRadius);
        polH_ = std::max(1, static_cast<int>(std::lround(2.0 * M_PI * maxRadius)));

        size_t cartCount = static_cast<size_t>(rows_) * cols_;
        size_t polarCount = static_cast<size_t>(polH_) * polW_;
        CUDA_CHECK(cudaMalloc(&d_cartesianIn_, cartCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_polar_, polarCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_median_, polarCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diff_, polarCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ringEstimatePolar_, polarCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ringEstimateCartesian_, cartCount * sizeof(float)));
    }

    ~Impl()
    {
        cudaFree(d_cartesianIn_);
        cudaFree(d_polar_);
        cudaFree(d_median_);
        cudaFree(d_diff_);
        cudaFree(d_ringEstimatePolar_);
        cudaFree(d_ringEstimateCartesian_);
    }
};

PolarRingCudaBackend::PolarRingCudaBackend(int rows, int cols, double maskRadiusRatio)
    : impl_(std::make_unique<Impl>(rows, cols, maskRadiusRatio))
{
}

PolarRingCudaBackend::~PolarRingCudaBackend() = default;

cv::Mat PolarRingCudaBackend::remove_ring(const cv::Mat& slice, double thresh, double threshMax, double threshMin,
                                           double thetaMinDeg, int ringWidth, bool wrapBoundary)
{
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> lock(impl.mutex_);

    CV_Assert(slice.type() == CV_32FC1);
    CV_Assert(slice.rows == impl.rows_ && slice.cols == impl.cols_);

    CUDA_CHECK(cudaMemcpy2D(impl.d_cartesianIn_, impl.cols_ * sizeof(float), slice.ptr<float>(0), slice.step,
                             impl.cols_ * sizeof(float), impl.rows_, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 polarGrid = grid2d(impl.polW_, impl.polH_, block);
    polarTransformKernel<<<polarGrid, block>>>(impl.d_cartesianIn_, impl.rows_, impl.cols_, impl.d_polar_,
                                                impl.polH_, impl.polW_, impl.centerX_, impl.centerY_,
                                                threshMax, threshMin);
    CUDA_CHECK(cudaGetLastError());

    const int m_rad = 2 * ringWidth + 1;
    radialMedianKernel<<<polarGrid, block>>>(impl.d_polar_, impl.polH_, impl.polW_, m_rad, impl.d_median_);
    CUDA_CHECK(cudaGetLastError());

    const size_t polarCount = static_cast<size_t>(impl.polH_) * impl.polW_;
    dim3 block1d(256);
    dim3 polarGrid1d(static_cast<unsigned>((polarCount + block1d.x - 1) / block1d.x));
    diffThresholdKernel<<<polarGrid1d, block1d>>>(impl.d_polar_, impl.d_median_, polarCount, thresh, impl.d_diff_);
    CUDA_CHECK(cudaGetLastError());

    const int m_azi = static_cast<int>(std::floor(impl.polH_ / 360.0 * thetaMinDeg));
    azimuthalMeanKernel<<<polarGrid, block>>>(impl.d_diff_, impl.polH_, impl.polW_, m_azi, wrapBoundary,
                                               impl.d_ringEstimatePolar_);
    CUDA_CHECK(cudaGetLastError());

    dim3 cartGrid = grid2d(impl.cols_, impl.rows_, block);
    inversePolarTransformKernel<<<cartGrid, block>>>(impl.d_ringEstimatePolar_, impl.polH_, impl.polW_,
                                                      impl.rows_, impl.cols_, impl.centerX_, impl.centerY_,
                                                      impl.d_ringEstimateCartesian_);
    CUDA_CHECK(cudaGetLastError());

    const size_t cartCount = static_cast<size_t>(impl.rows_) * impl.cols_;
    dim3 cartGrid1d(static_cast<unsigned>((cartCount + block1d.x - 1) / block1d.x));
    subtractKernel<<<cartGrid1d, block1d>>>(impl.d_cartesianIn_, impl.d_ringEstimateCartesian_, cartCount);
    CUDA_CHECK(cudaGetLastError());

    cv::Mat corrected(impl.rows_, impl.cols_, CV_32FC1);
    CUDA_CHECK(cudaMemcpy(corrected.ptr<float>(0), impl.d_cartesianIn_, cartCount * sizeof(float),
                           cudaMemcpyDeviceToHost));
    return corrected;
}
