#include "fbp_accelerate_backend.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace {

// Frequency index in FFT sample order: 0, 1, ..., N/2, -(N/2-1), ..., -1.
inline int fft_freq_index(int k, int N)
{
    return (k <= N / 2) ? k : k - N;
}

} // namespace

FbpAccelerateBackend::FbpAccelerateBackend(int n_detectors, int filter_size, const std::vector<float>& filter)
    : n_detectors_(n_detectors), filter_size_(filter_size), filter_(filter)
{
    log2_filter_size_ = static_cast<vDSP_Length>(std::log2(static_cast<double>(filter_size_)));
    fft_setup_ = vDSP_create_fftsetup(log2_filter_size_, kFFTRadix2);
    if (!fft_setup_)
        throw std::runtime_error("FbpAccelerateBackend: vDSP_create_fftsetup failed");
}

FbpAccelerateBackend::~FbpAccelerateBackend()
{
    vDSP_destroy_fftsetup(fft_setup_);
}

void FbpAccelerateBackend::filter_row(const float* in, float* out) const
{
    const int N = filter_size_;
    std::vector<float> real(N, 0.0f), imag(N, 0.0f);
    std::copy(in, in + n_detectors_, real.begin());

    DSPSplitComplex data{ real.data(), imag.data() };
    vDSP_fft_zip(fft_setup_, &data, 1, log2_filter_size_, kFFTDirection_Forward);

    vDSP_vmul(data.realp, 1, filter_.data(), 1, data.realp, 1, N);
    vDSP_vmul(data.imagp, 1, filter_.data(), 1, data.imagp, 1, N);

    vDSP_fft_zip(fft_setup_, &data, 1, log2_filter_size_, kFFTDirection_Inverse);
    // filter_ is already pre-scaled by 1/N (see header), so no separate scale pass here.

    std::copy(real.begin(), real.begin() + n_detectors_, out);
}

void FbpAccelerateBackend::shiftSinogram(float* sinogram, int n_angles, std::size_t rowStrideBytes, double shift)
{
    if (shift == 0.0)
        return;

    const int N = filter_size_;
    std::vector<float> phaseReal(N), phaseImag(N);
    for (int k = 0; k < N; ++k) {
        int freq_k = fft_freq_index(k, N);
        double angle = -2.0 * M_PI * freq_k * shift / static_cast<double>(N);
        phaseReal[k] = static_cast<float>(std::cos(angle));
        phaseImag[k] = static_cast<float>(std::sin(angle));
    }

    std::vector<float> real(N), imag(N), newReal(N), newImag(N);
    auto* base = reinterpret_cast<unsigned char*>(sinogram);
    for (int row = 0; row < n_angles; ++row) {
        float* rowPtr = reinterpret_cast<float*>(base + static_cast<std::size_t>(row) * rowStrideBytes);

        std::fill(real.begin(), real.end(), 0.0f);
        std::fill(imag.begin(), imag.end(), 0.0f);
        std::copy(rowPtr, rowPtr + n_detectors_, real.begin());

        DSPSplitComplex data{ real.data(), imag.data() };
        vDSP_fft_zip(fft_setup_, &data, 1, log2_filter_size_, kFFTDirection_Forward);

        for (int k = 0; k < N; ++k) {
            newReal[k] = real[k] * phaseReal[k] - imag[k] * phaseImag[k];
            newImag[k] = real[k] * phaseImag[k] + imag[k] * phaseReal[k];
        }

        DSPSplitComplex data2{ newReal.data(), newImag.data() };
        vDSP_fft_zip(fft_setup_, &data2, 1, log2_filter_size_, kFFTDirection_Inverse);

        float scale = 1.0f / static_cast<float>(N);
        vDSP_vsmul(data2.realp, 1, &scale, data2.realp, 1, N);

        std::copy(newReal.begin(), newReal.begin() + n_detectors_, rowPtr);
    }
}

void FbpAccelerateBackend::reconstructSlice(const float* sinogram, int n_angles, std::size_t rowStrideBytes,
                                             const std::vector<double>& angles_rad, double circ_mask_ratio,
                                             float* outSlice)
{
    const auto* base = reinterpret_cast<const unsigned char*>(sinogram);
    auto rowPtr = [&](int a) {
        return reinterpret_cast<const float*>(base + static_cast<std::size_t>(a) * rowStrideBytes);
    };

    // 1. Filter every projection row (windowed ramp).
    std::vector<float> filtered(static_cast<std::size_t>(n_angles) * n_detectors_);
    for (int a = 0; a < n_angles; ++a)
        filter_row(rowPtr(a), filtered.data() + static_cast<std::size_t>(a) * n_detectors_);

    // 2. Backproject: f(x,y) = dtheta * sum_i Q_i(t), t = x*cos(theta_i) + y*sin(theta_i).
    const int N = n_detectors_;
    const double cx = N / 2.0;
    const double cy = N / 2.0;
    const double det_center = N / 2.0;
    const double dtheta = (n_angles > 1) ? (angles_rad[1] - angles_rad[0]) : M_PI / n_angles;

    std::vector<double> cosA(n_angles), sinA(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        cosA[a] = std::cos(angles_rad[a]);
        sinA[a] = std::sin(angles_rad[a]);
    }

    std::fill(outSlice, outSlice + static_cast<std::size_t>(N) * N, 0.0f);
    for (int y = 0; y < N; ++y) {
        float* outRow = outSlice + static_cast<std::size_t>(y) * N;
        double yc = y - cy;
        for (int x = 0; x < N; ++x) {
            double xc = x - cx;
            double sum = 0.0;
            for (int a = 0; a < n_angles; ++a) {
                double t = xc * cosA[a] + yc * sinA[a] + det_center;
                int t0 = static_cast<int>(std::floor(t));
                if (t0 < 0 || t0 + 1 >= N)
                    continue;
                double frac = t - t0;
                const float* frow = filtered.data() + static_cast<std::size_t>(a) * n_detectors_;
                sum += frow[t0] * (1.0 - frac) + frow[t0 + 1] * frac;
            }
            outRow[x] = static_cast<float>(sum * dtheta);
        }
    }

    // 3. Circular mask (ports tomopy.circ_mask).
    const double radius = circ_mask_ratio * (N / 2.0);
    const double radius2 = radius * radius;
    for (int y = 0; y < N; ++y) {
        float* outRow = outSlice + static_cast<std::size_t>(y) * N;
        double yc = y - cy;
        for (int x = 0; x < N; ++x) {
            double xc = x - cx;
            if (xc * xc + yc * yc > radius2)
                outRow[x] = 0.0f;
        }
    }
}
