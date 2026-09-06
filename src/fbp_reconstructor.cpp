#include "fbp_reconstructor.h"
#include "fbp_cuda_backend.h"
#include <cmath>
#include <stdexcept>

namespace {

int next_pow2(int minValue)
{
    int size = 1;
    while (size < minValue)
        size <<= 1;
    return size;
}

// Frequency index in FFT sample order: 0, 1, ..., N/2, -(N/2-1), ..., -1.
inline int fft_freq_index(int k, int N)
{
    return (k <= N / 2) ? k : k - N;
}

// Apodization window as a function of normalized frequency f in [-1, 1]
// (f = 0 at DC, f = +-1 at Nyquist) - Kak & Slaney convention. Cross-checked
// against the original hand-derived Hamming formula (window as a function of
// FFT-shifted array position m/(N-1)) to confirm the two parametrizations
// agree exactly at f=0 (peak ~1.0) and f=+-1 (taper ~0.08).
double apodization_window(FbpFilterType type, double f)
{
    switch (type) {
    case FbpFilterType::RamLak:
        return 1.0;
    case FbpFilterType::SheppLogan: {
        double x = M_PI * f / 2.0;
        return (f == 0.0) ? 1.0 : std::sin(x) / x;
    }
    case FbpFilterType::Cosine:
        return std::cos(M_PI * f / 2.0);
    case FbpFilterType::Hamming:
        return 0.54 + 0.46 * std::cos(M_PI * f);
    case FbpFilterType::Hann:
        return 0.5 + 0.5 * std::cos(M_PI * f);
    }
    return 1.0;
}

} // namespace

FbpReconstructor::FbpReconstructor(int n_detectors, FbpFilterType filterType)
    : n_detectors_(n_detectors)
{
    filter_size_ = next_pow2(2 * n_detectors_);

    // Windowed ramp filter, built directly in the frequency domain: ramp[k] =
    // 2*|freq_k|/N grows linearly with frequency (the ideal ramp response);
    // the apodization window tapers the noise-amplifying high frequencies
    // while leaving low frequencies close to untouched (RamLak = no taper at
    // all, i.e. the unwindowed ramp). Pre-scaled by 1/N so the GPU's batched
    // inverse FFT needs no separate scale pass.
    const int N = filter_size_;
    std::vector<float> filter(N);
    for (int k = 0; k < N; ++k) {
        int freq_k = fft_freq_index(k, N);
        double ramp = 2.0 * std::abs(freq_k) / static_cast<double>(N);
        double f_norm = freq_k / (N / 2.0);
        double window = apodization_window(filterType, f_norm);
        filter[k] = static_cast<float>(ramp * window / N);
    }

    cuda_ = std::make_unique<FbpCudaBackend>(n_detectors_, filter_size_, filter);
}

FbpReconstructor::~FbpReconstructor() = default;

void FbpReconstructor::shift_sinogram(cv::Mat& sinogram, double shift) const
{
    if (shift == 0.0)
        return;

    CV_Assert(sinogram.type() == CV_32FC1);
    CV_Assert(sinogram.cols == n_detectors_);

    cuda_->shiftSinogram(sinogram.ptr<float>(0), sinogram.rows, sinogram.step, shift);
}

cv::Mat FbpReconstructor::reconstruct_slice(const cv::Mat& sinogram, const std::vector<double>& angles_rad,
                                             double circ_mask_ratio) const
{
    CV_Assert(sinogram.type() == CV_32FC1);
    CV_Assert(sinogram.cols == n_detectors_);
    const int n_angles = sinogram.rows;
    if (static_cast<int>(angles_rad.size()) != n_angles)
        throw std::invalid_argument("FbpReconstructor::reconstruct_slice: angles_rad size mismatch");

    cv::Mat recon(n_detectors_, n_detectors_, CV_32FC1);
    cuda_->reconstructSlice(sinogram.ptr<float>(0), n_angles, sinogram.step, angles_rad,
                             circ_mask_ratio, recon.ptr<float>(0));
    return recon;
}
