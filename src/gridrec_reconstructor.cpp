#include "gridrec_reconstructor.h"
#if defined(__APPLE__)
#include "gridrec_accelerate_backend.h"
#else
#include "gridrec_cuda_backend.h"
#endif
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

// Mirrors fbp_reconstructor.cpp's apodization_window() exactly (kept as a
// separate copy since that one is private to fbp_reconstructor.cpp's
// anonymous namespace) - same Kak & Slaney windows applied to the same ramp.
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

// Modified Bessel function of the first kind, order 0. Abramowitz & Stegun
// 9.8.1-9.8.2 polynomial approximation (~1e-7 relative error), used for the
// Kaiser-Bessel gridding kernel and its deapodization correction below.
double bessel_i0(double x)
{
    double ax = std::abs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
               + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    }
    double t = 3.75 / ax;
    return (std::exp(ax) / std::sqrt(ax))
           * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
              + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
              + t * (-0.01647633 + t * 0.00392377))))))));
}

// Kaiser-Bessel convolution weight at offset `d` (grid pixels) from a sample,
// kernel support `width` (total, i.e. nonzero for |d| <= width/2).
double kb_weight(double d, double width, double beta)
{
    double half = width / 2.0;
    if (std::abs(d) >= half)
        return 0.0;
    double r = d / half;
    return bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / bessel_i0(beta);
}

} // namespace

GridrecReconstructor::GridrecReconstructor(int n_detectors, FbpFilterType filterType)
    : n_detectors_(n_detectors)
{
    grid_size_ = next_pow2(2 * n_detectors_);
    const int N = grid_size_;

    // Windowed ramp filter - same math as FbpReconstructor's, but NOT
    // pre-scaled by 1/N: gridrec's overall amplitude correction happens once
    // at the end (see the backends' `scale` factor), not per filtered row.
    std::vector<float> filter(N);
    for (int k = 0; k < N; ++k) {
        int freq_k = fft_freq_index(k, N);
        double ramp = 2.0 * std::abs(freq_k) / static_cast<double>(N);
        double f_norm = freq_k / (N / 2.0);
        filter[k] = static_cast<float>(ramp * apodization_window(filterType, f_norm));
    }

    // Projections are zero-padded starting at array index 0, but the
    // Fourier-slice-theorem mapping needs the origin at the rotation axis
    // (the detector midpoint). DFT shift theorem: shifting the space-domain
    // origin by -det_center multiplies the spectrum by exp(+2*pi*i*freq_k*
    // det_center/N). Real-space FBP gets this for free via its `t = ... +
    // det_center` indexing; gridrec must correct for it explicitly, or the
    // Fourier-slice angle mapping is wrong and the reconstruction is
    // meaningless.
    const double det_center = n_detectors_ / 2.0;
    std::vector<float> centerPhaseReal(N), centerPhaseImag(N);
    for (int k = 0; k < N; ++k) {
        int freq_k = fft_freq_index(k, N);
        double angle = 2.0 * M_PI * freq_k * det_center / static_cast<double>(N);
        centerPhaseReal[k] = static_cast<float>(std::cos(angle));
        centerPhaseImag[k] = static_cast<float>(std::sin(angle));
    }

    // Small (4-pixel-wide) Kaiser-Bessel gridding kernel. beta from Beatty et
    // al. 2005 eq. 5, tuned for the oversampling ratio grid_size_/n_detectors_
    // (~2x here since grid_size_ = next_pow2(2*n_detectors_)).
    const double kb_width = 4.0;
    double os = static_cast<double>(grid_size_) / n_detectors_;
    double kb_beta = M_PI * std::sqrt((kb_width / os) * (kb_width / os) * (os - 0.5) * (os - 0.5) - 0.8);

    // Deapodization: 1 / (this kernel's own discrete Fourier transform),
    // computed directly as an O(N^2) DFT sum rather than via a platform FFT
    // library - a one-time, N up to a few thousand cost at construction, and
    // it keeps this facade (shared between the Accelerate and CUDA backends)
    // free of any platform-specific FFT dependency. Building it from the
    // literal discrete kernel used for gridding (rather than the closed-form
    // KB apodization-correction formula, which is sensitive to a spatial-
    // coordinate normalization that's easy to get subtly wrong) is
    // self-consistent by construction.
    std::vector<double> kernFreq(N);
    for (int k = 0; k < N; ++k)
        kernFreq[k] = kb_weight(fft_freq_index(k, N), kb_width, kb_beta);

    std::vector<float> deapod_1d(N);
    for (int i = 0; i < N; ++i) {
        int x = i - N / 2;
        double re = 0.0;
        for (int k = 0; k < N; ++k) {
            double angle = 2.0 * M_PI * k * x / static_cast<double>(N);
            re += kernFreq[k] * std::cos(angle);
        }
        deapod_1d[i] = static_cast<float>(1.0 / re);
    }

#if defined(__APPLE__)
    backend_ = std::make_unique<GridrecAccelerateBackend>(n_detectors_, grid_size_, filter, kb_width, kb_beta,
                                                            deapod_1d, centerPhaseReal, centerPhaseImag);
#else
    backend_ = std::make_unique<GridrecCudaBackend>(n_detectors_, grid_size_, filter, kb_width, kb_beta,
                                                      deapod_1d, centerPhaseReal, centerPhaseImag);
#endif
}

GridrecReconstructor::~GridrecReconstructor() = default;

void GridrecReconstructor::shift_sinogram(cv::Mat& sinogram, double shift) const
{
    if (shift == 0.0)
        return;

    CV_Assert(sinogram.type() == CV_32FC1);
    CV_Assert(sinogram.cols == n_detectors_);

    backend_->shiftSinogram(sinogram.ptr<float>(0), sinogram.rows, sinogram.step, shift);
}

cv::Mat GridrecReconstructor::reconstruct_slice(const cv::Mat& sinogram, const std::vector<double>& angles_rad,
                                                 double circ_mask_ratio) const
{
    CV_Assert(sinogram.type() == CV_32FC1);
    CV_Assert(sinogram.cols == n_detectors_);
    const int n_angles = sinogram.rows;
    if (static_cast<int>(angles_rad.size()) != n_angles)
        throw std::invalid_argument("GridrecReconstructor::reconstruct_slice: angles_rad size mismatch");

    cv::Mat recon(n_detectors_, n_detectors_, CV_32FC1);
    backend_->reconstructSlice(sinogram.ptr<float>(0), n_angles, sinogram.step, angles_rad,
                                circ_mask_ratio, recon.ptr<float>(0));
    return recon;
}
