#include "gridrec_accelerate_backend.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace {

// Frequency index in FFT sample order: 0, 1, ..., N/2, -(N/2-1), ..., -1.
inline int fft_freq_index(int k, int N)
{
    return (k <= N / 2) ? k : k - N;
}

// Modified Bessel function of the first kind, order 0 - same Abramowitz &
// Stegun approximation as gridrec_reconstructor.cpp's copy (kept separate
// since that one is private to its own anonymous namespace).
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

GridrecAccelerateBackend::GridrecAccelerateBackend(int n_detectors, int grid_size, const std::vector<float>& filter,
                                                     double kb_width, double kb_beta,
                                                     const std::vector<float>& deapod_1d,
                                                     const std::vector<float>& centerPhaseReal,
                                                     const std::vector<float>& centerPhaseImag)
    : n_detectors_(n_detectors), grid_size_(grid_size), filter_(filter), kb_width_(kb_width), kb_beta_(kb_beta),
      deapod_1d_(deapod_1d), centerPhaseReal_(centerPhaseReal), centerPhaseImag_(centerPhaseImag)
{
    log2_grid_ = static_cast<vDSP_Length>(std::log2(static_cast<double>(grid_size_)));
    fft_setup_ = vDSP_create_fftsetup(log2_grid_, kFFTRadix2);
    if (!fft_setup_)
        throw std::runtime_error("GridrecAccelerateBackend: vDSP_create_fftsetup failed");
}

GridrecAccelerateBackend::~GridrecAccelerateBackend()
{
    vDSP_destroy_fftsetup(fft_setup_);
}

void GridrecAccelerateBackend::filterRowFreq(const float* in, float* outReal, float* outImag) const
{
    const int N = grid_size_;
    std::fill(outReal, outReal + N, 0.0f);
    std::fill(outImag, outImag + N, 0.0f);
    std::copy(in, in + n_detectors_, outReal);

    DSPSplitComplex data{ outReal, outImag };
    vDSP_fft_zip(fft_setup_, &data, 1, log2_grid_, kFFTDirection_Forward);

    vDSP_vmul(outReal, 1, filter_.data(), 1, outReal, 1, N);
    vDSP_vmul(outImag, 1, filter_.data(), 1, outImag, 1, N);

    // Re-center the origin onto the rotation axis (see GridrecReconstructor's
    // centerPhase comment) - without this the gridded spectrum is
    // angle-mapped around the wrong point and the reconstruction is
    // meaningless.
    for (int k = 0; k < N; ++k) {
        float a = outReal[k], b = outImag[k];
        float cp = centerPhaseReal_[k], sp = centerPhaseImag_[k];
        outReal[k] = a * cp - b * sp;
        outImag[k] = a * sp + b * cp;
    }
}

void GridrecAccelerateBackend::shiftSinogram(float* sinogram, int n_angles, std::size_t rowStrideBytes, double shift)
{
    if (shift == 0.0)
        return;

    const int N = grid_size_;
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
        vDSP_fft_zip(fft_setup_, &data, 1, log2_grid_, kFFTDirection_Forward);

        for (int k = 0; k < N; ++k) {
            newReal[k] = real[k] * phaseReal[k] - imag[k] * phaseImag[k];
            newImag[k] = real[k] * phaseImag[k] + imag[k] * phaseReal[k];
        }

        DSPSplitComplex data2{ newReal.data(), newImag.data() };
        vDSP_fft_zip(fft_setup_, &data2, 1, log2_grid_, kFFTDirection_Inverse);

        float scale = 1.0f / static_cast<float>(N);
        vDSP_vsmul(data2.realp, 1, &scale, data2.realp, 1, N);

        std::copy(newReal.begin(), newReal.begin() + n_detectors_, rowPtr);
    }
}

void GridrecAccelerateBackend::reconstructSlice(const float* sinogram, int n_angles, std::size_t rowStrideBytes,
                                                 const std::vector<double>& angles_rad, double circ_mask_ratio,
                                                 float* outSlice)
{
    const auto* base = reinterpret_cast<const unsigned char*>(sinogram);
    auto rowPtr = [&](int a) {
        return reinterpret_cast<const float*>(base + static_cast<std::size_t>(a) * rowStrideBytes);
    };

    const int N = grid_size_;
    const double dtheta = (n_angles > 1) ? (angles_rad[1] - angles_rad[0]) : M_PI / n_angles;

    // 2D frequency-domain grid, laid out in standard FFT order (index 0 =
    // DC) on both axes, so scattered samples wrap via plain modulo below
    // with no separate fftshift needed on the frequency side.
    std::vector<float> gridReal(static_cast<std::size_t>(N) * N, 0.0f);
    std::vector<float> gridImag(static_cast<std::size_t>(N) * N, 0.0f);

    const int kr = static_cast<int>(std::ceil(kb_width_ / 2.0));
    std::vector<float> rowReal(N), rowImag(N);

    for (int a = 0; a < n_angles; ++a) {
        filterRowFreq(rowPtr(a), rowReal.data(), rowImag.data());

        double c = std::cos(angles_rad[a]);
        double s = std::sin(angles_rad[a]);

        // Fourier slice theorem: FT_1D(filtered projection)(freq_k) equals
        // the 2D spectrum sampled along the line through the origin at this
        // angle. Scatter each sample onto nearby grid cells with a 2D
        // separable Kaiser-Bessel weight (gridding convolution).
        for (int k = 0; k < N; ++k) {
            int freq_k = fft_freq_index(k, N);
            double u = freq_k * c;
            double v = freq_k * s;

            int u0 = static_cast<int>(std::floor(u));
            int v0 = static_cast<int>(std::floor(v));

            for (int dv = -kr + 1; dv <= kr; ++dv) {
                double wv = kb_weight(v0 + dv - v, kb_width_, kb_beta_);
                if (wv == 0.0)
                    continue;
                int iv = ((v0 + dv) % N + N) % N;
                for (int du = -kr + 1; du <= kr; ++du) {
                    double wu = kb_weight(u0 + du - u, kb_width_, kb_beta_);
                    if (wu == 0.0)
                        continue;
                    int iu = ((u0 + du) % N + N) % N;
                    double w = wu * wv;
                    std::size_t idx = static_cast<std::size_t>(iv) * N + iu;
                    gridReal[idx] += static_cast<float>(w * rowReal[k]);
                    gridImag[idx] += static_cast<float>(w * rowImag[k]);
                }
            }
        }
    }

    // 2D inverse FFT as two 1D passes (rows, then columns) with the same
    // vDSP_fft_zip primitive filterRowFreq/shiftSinogram already use, rather
    // than vDSP_fft2d_zip's less-obvious dual-stride convention.
    for (int y = 0; y < N; ++y) {
        DSPSplitComplex row{ gridReal.data() + static_cast<std::size_t>(y) * N,
                              gridImag.data() + static_cast<std::size_t>(y) * N };
        vDSP_fft_zip(fft_setup_, &row, 1, log2_grid_, kFFTDirection_Inverse);
    }
    for (int x = 0; x < N; ++x) {
        DSPSplitComplex col{ gridReal.data() + x, gridImag.data() + x };
        vDSP_fft_zip(fft_setup_, &col, N, log2_grid_, kFFTDirection_Inverse);
    }

    // The 2D IFFT above is unnormalized (gain N^2), but deapod_1d_ was built
    // from an also-unnormalized DFT sum (see GridrecReconstructor's
    // comment), so that gain is already cancelled once by the
    // deapod_1d_.x * deapod_1d_.y multiply below - only one remaining factor
    // of N needs correcting here, not N^2 (empirically confirmed against
    // FbpReconstructor's peak value on a centered point-source phantom,
    // matching to <1%).
    float scale = static_cast<float>(dtheta / static_cast<double>(N));
    vDSP_vsmul(gridReal.data(), 1, &scale, gridReal.data(), 1, static_cast<vDSP_Length>(gridReal.size()));

    // The inverse FFT above was fed a grid in standard FFT order, so its
    // output is also in that order: object center (x=0) sits at array index
    // 0 and wraps to the high end for negative coordinates. fftshift it back
    // to a natural layout (center at N/2) before cropping and deapodizing.
    const int half = N / 2;
    const int off = (N - n_detectors_) / 2;
    for (int y = 0; y < n_detectors_; ++y) {
        int gy = (off + y + half) % N;
        float* outRow = outSlice + static_cast<std::size_t>(y) * n_detectors_;
        float deapodY = deapod_1d_[off + y];
        for (int x = 0; x < n_detectors_; ++x) {
            int gx = (off + x + half) % N;
            outRow[x] = gridReal[static_cast<std::size_t>(gy) * N + gx] * deapodY * deapod_1d_[off + x];
        }
    }

    // Circular mask (ports tomopy.circ_mask), same as FbpAccelerateBackend.
    const double cx = n_detectors_ / 2.0;
    const double cy = n_detectors_ / 2.0;
    const double radius = circ_mask_ratio * (n_detectors_ / 2.0);
    const double radius2 = radius * radius;
    for (int y = 0; y < n_detectors_; ++y) {
        float* outRow = outSlice + static_cast<std::size_t>(y) * n_detectors_;
        double yc = y - cy;
        for (int x = 0; x < n_detectors_; ++x) {
            double xc = x - cx;
            if (xc * xc + yc * yc > radius2)
                outRow[x] = 0.0f;
        }
    }
}
