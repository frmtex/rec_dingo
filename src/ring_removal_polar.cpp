#include "ring_removal_polar.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace {

// Runs body(i) for every i in [0, count). When `parallel` is true, splits the work across
// hardware_concurrency() threads - each slice/preview call is one top-level invocation of
// remove_ring, so a fresh thread pool per loop (rather than a persistent one) is negligible
// overhead next to the per-row/column work it's covering. `parallel` must be false when the
// caller (post_process_worker.cpp) is itself already running this call inside its own per-file
// thread pool: nesting two independent thread pools oversubscribes the machine (e.g. 12 outer x
// 12 inner = 144 threads fighting over 12 cores), which is far slower than either level alone.
// body must only touch index-i-local state (or synchronize itself) - the loops here all write to
// disjoint rows/columns of their output, so no synchronization is needed.
template <typename Func>
void parallelFor(int count, bool parallel, const Func& body)
{
    unsigned int numThreads = parallel ? std::max(1u, std::thread::hardware_concurrency()) : 1u;
    numThreads = std::min(numThreads, static_cast<unsigned int>(std::max(1, count)));
    if (numThreads <= 1) {
        for (int i = 0; i < count; ++i)
            body(i);
        return;
    }

    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t) {
        pool.emplace_back([&]() {
            for (;;) {
                int i = next.fetch_add(1);
                if (i >= count)
                    return;
                body(i);
            }
        });
    }
    for (auto& th : pool)
        th.join();
}

// Kernel radius for the inner/middle/outer thirds of the radius range - finer near the
// center, coarser further out. `design` is 2*ringWidth+1 for the radial median filter, or
// floor(polH/360*thetaMinDeg) for the azimuthal mean filter.
int bandedKernelRadius(int col, int polW, int design)
{
    int b1 = polW / 3;
    int b2 = (2 * polW) / 3;
    if (col < b1) return design / 3;
    if (col < b2) return (2 * design) / 3;
    return design;
}

// Nearest-neighbor Cartesian -> polar resample, centered on (centerX, centerY). Radius maps to
// column, angle to row. Samples are taken at the angular bin *center* (a half-bin offset), and
// clamped to [threshMin, threshMax].
cv::Mat polarTransform(const cv::Mat& img, double centerX, double centerY, int polH, int polW,
                        double threshMax, double threshMin, bool parallel)
{
    cv::Mat polar(polH, polW, CV_32FC1);
    const int rows = img.rows, cols = img.cols;
    parallelFor(polH, parallel, [&](int a) {
        double theta = a * 2.0 * M_PI / polH + M_PI / polH;
        double c = std::cos(theta), s = std::sin(theta);
        float* dst = polar.ptr<float>(a);
        for (int r = 0; r < polW; ++r) {
            int xi = static_cast<int>(std::lround(r * c + centerX));
            int yi = static_cast<int>(std::lround(r * s + centerY));
            xi = std::clamp(xi, 0, cols - 1);
            yi = std::clamp(yi, 0, rows - 1);
            float val = img.at<float>(yi, xi);
            dst[r] = static_cast<float>(std::clamp(static_cast<double>(val), threshMin, threshMax));
        }
    });
    return polar;
}

// Inverse of polarTransform: for each Cartesian pixel, look up its polar-space location (undoing
// the same half-bin angular offset) and sample the polar buffer, nearest-neighbor. Pixels outside
// the polar buffer's radius range (the disk actually covered by polarTransform) are 0.
cv::Mat inversePolarTransform(const cv::Mat& polar, double centerX, double centerY, int rows, int cols,
                               bool parallel)
{
    const int polH = polar.rows, polW = polar.cols;
    cv::Mat out = cv::Mat::zeros(rows, cols, CV_32FC1);
    parallelFor(rows, parallel, [&](int y) {
        double dy = y - centerY;
        float* dst = out.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            double dx = x - centerX;
            double theta = std::atan2(dy, dx) - M_PI / polH;
            if (theta < 0.0)
                theta += 2.0 * M_PI;
            int polRow = static_cast<int>(std::lround(theta * polH / (2.0 * M_PI)));
            if (polRow >= polH) polRow -= polH;
            if (polRow < 0) polRow += polH;
            int polCol = static_cast<int>(std::lround(std::sqrt(dx * dx + dy * dy)));
            if (polRow >= 0 && polRow < polH && polCol >= 0 && polCol < polW)
                dst[x] = polar.at<float>(polRow, polCol);
        }
    });
    return out;
}

// Median-filters `polar` across columns (radius) independently for each row (angle), with a
// kernel radius that grows in three radius bands (see bandedKernelRadius). A window index that
// goes negative crosses through the rotation axis (r=0): the same physical point at radius |idx|
// lies at the opposite angle, so it's sampled from row (a + polH/2) mod polH, column -idx, rather
// than just being clamped or reflected in place.
cv::Mat radialMedianFilter(const cv::Mat& polar, int design, bool parallel)
{
    const int polH = polar.rows, polW = polar.cols;
    cv::Mat out(polH, polW, CV_32FC1);
    parallelFor(polH, parallel, [&](int a) {
        // Declared per-row (not hoisted above the loop) since each thread needs its own -
        // sharing one would race across concurrent rows.
        std::vector<float> window;
        const float* rowPtr = polar.ptr<float>(a);
        const float* oppositeRowPtr = polar.ptr<float>((a + polH / 2) % polH);
        float* dst = out.ptr<float>(a);
        for (int c = 0; c < polW; ++c) {
            int kr = bandedKernelRadius(c, polW, design);
            window.clear();
            for (int k = -kr; k <= kr; ++k) {
                int idx = c + k;
                if (idx < 0)
                    window.push_back(oppositeRowPtr[std::min(-idx, polW - 1)]);
                else if (idx >= polW)
                    window.push_back(rowPtr[polW - 1]);
                else
                    window.push_back(rowPtr[idx]);
            }
            size_t mid = window.size() / 2;
            std::nth_element(window.begin(), window.begin() + mid, window.end());
            dst[c] = window[mid];
        }
    });
    return out;
}

// Mean-filters `diff` across rows (angle) independently for each column (radius), with a kernel
// radius that grows in three radius bands. Boundary handling is circular (wrapBoundary) or mirror
// reflection otherwise. A pixel whose own value is exactly 0 (rejected by the threshold step
// upstream) stays 0 rather than picking up a neighbor-derived estimate - only pixels that survived
// thresholding get smoothed/reinforced into a ring estimate.
cv::Mat azimuthalMeanFilter(const cv::Mat& diff, int design, bool wrapBoundary, bool parallel)
{
    const int polH = diff.rows, polW = diff.cols;
    cv::Mat out(polH, polW, CV_32FC1);

    auto reflectIndex = [polH](int idx) {
        while (idx < 0 || idx >= polH) {
            if (idx < 0) idx = -idx - 1;
            if (idx >= polH) idx = 2 * polH - 1 - idx;
        }
        return idx;
    };

    parallelFor(polW, parallel, [&](int c) {
        int kr = bandedKernelRadius(c, polW, design);
        int windowSize = 2 * kr + 1;
        for (int a = 0; a < polH; ++a) {
            float self = diff.at<float>(a, c);
            if (self == 0.0f) {
                out.at<float>(a, c) = 0.0f;
                continue;
            }
            double sum = 0.0;
            for (int k = -kr; k <= kr; ++k) {
                int idx = a + k;
                if (wrapBoundary)
                    idx = ((idx % polH) + polH) % polH;
                else
                    idx = reflectIndex(idx);
                sum += diff.at<float>(idx, c);
            }
            out.at<float>(a, c) = static_cast<float>(sum / windowSize);
        }
    });
    return out;
}

} // namespace

cv::Mat PolarRingRemoval::remove_ring(const cv::Mat& slice, double thresh, double threshMax, double threshMin,
                                       double thetaMinDeg, int ringWidth, bool wrapBoundary, bool parallel,
                                       double maskRadiusRatio)
{
    CV_Assert(slice.type() == CV_32FC1);

    const double centerX = (slice.cols - 1) / 2.0;
    const double centerY = (slice.rows - 1) / 2.0;
    const int geometricMaxRadius = static_cast<int>(std::floor(
        std::min({ centerX, (slice.cols - 1) - centerX, centerY, (slice.rows - 1) - centerY })));
    const int maskRadius =
        static_cast<int>(std::floor(maskRadiusRatio * std::min(slice.rows, slice.cols) / 2.0));
    const int maxRadius = std::min(geometricMaxRadius, maskRadius);
    const int polW = std::max(1, maxRadius);
    const int polH = std::max(1, static_cast<int>(std::lround(2.0 * M_PI * maxRadius)));

    cv::Mat polar = polarTransform(slice, centerX, centerY, polH, polW, threshMax, threshMin, parallel);

    const int m_rad = 2 * ringWidth + 1;
    cv::Mat median = radialMedianFilter(polar, m_rad, parallel);

    cv::Mat diff(polH, polW, CV_32FC1);
    parallelFor(polH, parallel, [&](int a) {
        const float* p = polar.ptr<float>(a);
        const float* m = median.ptr<float>(a);
        float* d = diff.ptr<float>(a);
        for (int c = 0; c < polW; ++c) {
            float v = p[c] - m[c];
            d[c] = (std::abs(v) > thresh) ? 0.0f : v;
        }
    });

    const int m_azi = static_cast<int>(std::floor(polH / 360.0 * thetaMinDeg));
    cv::Mat ringEstimatePolar = azimuthalMeanFilter(diff, m_azi, wrapBoundary, parallel);

    cv::Mat ringEstimate =
        inversePolarTransform(ringEstimatePolar, centerX, centerY, slice.rows, slice.cols, parallel);

    cv::Mat corrected;
    cv::subtract(slice, ringEstimate, corrected);
    return corrected;
}

int PolarRingRemoval::centerExclusionRadius(bool waveletFilterEnabled, int maskInnerRadius, int maskOuterRadius)
{
    if (waveletFilterEnabled && maskOuterRadius > maskInnerRadius && maskOuterRadius > 0 && maskInnerRadius > 0)
        return maskInnerRadius;
    return 0;
}

void PolarRingRemoval::restore_center(const cv::Mat& original, cv::Mat& corrected, int innerRadius)
{
    if (innerRadius <= 0)
        return;
    CV_Assert(original.type() == CV_32FC1 && corrected.type() == CV_32FC1);
    CV_Assert(original.size() == corrected.size());

    const double centerX = (corrected.cols - 1) / 2.0;
    const double centerY = (corrected.rows - 1) / 2.0;
    const double r2 = static_cast<double>(innerRadius) * innerRadius;
    const int y0 = std::max(0, static_cast<int>(std::ceil(centerY - innerRadius)));
    const int y1 = std::min(corrected.rows - 1, static_cast<int>(std::floor(centerY + innerRadius)));
    for (int y = y0; y <= y1; ++y) {
        const double dy = y - centerY;
        const float* src = original.ptr<float>(y);
        float* dst = corrected.ptr<float>(y);
        for (int x = 0; x < corrected.cols; ++x) {
            const double dx = x - centerX;
            if (dx * dx + dy * dy < r2)
                dst[x] = src[x];
        }
    }
}
