#include "phase_contrast.h"
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// image:  real-valued transmission image, row-major, nx rows x ny cols,
//         already converted from uint16 to float (NOT log-corrected).
// nx, ny: must both be powers of two (pad beforehand if not).
// alpha:  Paganin filter strength (delta/mu * propagation distance).
// pix:    pixel size; leave at 1.0 if unknown.
//
// Returns: retrieved projected thickness, same nx*ny size as input.
std::vector<float> phase_contrast::phase_retrieval(const std::vector<float>& image,
                                   int nx, int ny,
                                   float alpha, float pix = 1.0f)
{
    auto is_pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!is_pow2(nx) || !is_pow2(ny)) {
        throw std::invalid_argument(
            "phase_retrieval: nx and ny must be powers of two "
            "(pad the image before calling)");
    }
    if ((size_t)nx * (size_t)ny != image.size()) {
        throw std::invalid_argument("phase_retrieval: image size != nx*ny");
    }

    const size_t n = (size_t)nx * (size_t)ny;
    const vDSP_Length log2nx = (vDSP_Length)std::log2((double)nx);
    const vDSP_Length log2ny = (vDSP_Length)std::log2((double)ny);

    FFTSetup setup = vDSP_create_fftsetup(std::max(log2nx, log2ny), kFFTRadix2);
    if (!setup) throw std::runtime_error("vDSP_create_fftsetup failed");

    std::vector<float> real(image.begin(), image.end());
    std::vector<float> imag(n, 0.0f);
    DSPSplitComplex data{ real.data(), imag.data() };

    // --- forward 2D FFT, in place ---
    // Verify log2nx/log2ny ordering matches your data layout (caveat #2).
    vDSP_fft2d_zip(setup, &data, 1, 0, log2nx, log2ny, kFFTDirection_Forward);

    // --- build the real-valued Paganin filter, FFT-frequency ordered ---
    std::vector<float> filter(n);
    for (int i = 0; i < nx; ++i) {
        int kxi = (i <= nx / 2) ? i : i - nx;          // FFT freq order
        float kx = 2.0f * (float)M_PI * kxi / (nx * pix);
        float coskx = std::cos(kx * pix);
        for (int j = 0; j < ny; ++j) {
            int kyi = (j <= ny / 2) ? j : j - ny;
            float ky = 2.0f * (float)M_PI * kyi / (ny * pix);
            float cosky = std::cos(ky * pix);
            filter[(size_t)i * ny + j] =
                1.0f - (2.0f * alpha / (pix * pix)) * (coskx + cosky - 2.0f);
        }
    }

    // --- fim / filter  (complex / real -> divide both planes) ---
    // vDSP_vdiv(A, ..., B, ..., C, ...) computes C = B / A.
    vDSP_vdiv(filter.data(), 1, data.realp, 1, data.realp, 1, n);
    vDSP_vdiv(filter.data(), 1, data.imagp, 1, data.imagp, 1, n);

    // --- inverse 2D FFT, in place ---
    vDSP_fft2d_zip(setup, &data, 1, 0, log2nx, log2ny, kFFTDirection_Inverse);

    // Undo Accelerate's unnormalized round-trip scaling (caveat #3).
    float scale = 1.0f / (float)n;
    vDSP_vsmul(data.realp, 1, &scale, data.realp, 1, n);
    vDSP_vsmul(data.imagp, 1, &scale, data.imagp, 1, n);

    // --- magnitude, then -log ---
    std::vector<float> mag(n);
    vDSP_zvabs(&data, 1, mag.data(), 1, n);

    std::vector<float> phret(n);
    int nInt = (int)n;
    vvlogf(phret.data(), mag.data(), &nInt);        // phret = log(mag)
    float negOne = -1.0f, zero = 0.0f;
    vDSP_vsmsa(phret.data(), 1, &negOne, &zero, phret.data(), 1, n); // *-1

    vDSP_destroy_fftsetup(setup);
    return phret;
}