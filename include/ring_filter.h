#ifndef RING_FILTER_H
#define RING_FILTER_H
#include <opencv2/opencv.hpp>

// Ring-artifact suppression for a single sinogram (n_angles x n_detectors),
// using the combined wavelet-Fourier method of Munch, Trtik, Marone &
// Stampanoni (Opt. Express 17(10), 8567-8591, 2009): decompose with a
// Daubechies wavelet, damp low-angular-frequency content in the
// vertical-detail subband (the subband where angle-constant, detector-
// localized stripes concentrate their energy) in the Fourier domain, then
// reconstruct. Ports _ring_removal_wavelets_sinogram from the Python
// reconstruction pipeline.
//
// Daubechies coefficients (order 1-8) are validated against the standard
// orthogonal QMF conditions the first time they're used; a transcription
// error in a coefficient table throws immediately instead of silently
// degrading results.
class RingFilter
{
public:
    // sinogram: CV_32FC1, modified in place.
    // level: wavelet decomposition depth.
    // sigma: damping strength (larger = stronger stripe suppression).
    // order: Daubechies wavelet order, 1-8 (db1 = Haar .. db8).
    // pad: padding (mean-padded rows, edge-padded columns) added before decomposition to
    //      keep boundary effects away from the real data, matching the Python default of 200.
    // maskInnerRadius/maskOuterRadius: if maskOuterRadius > maskInnerRadius (and > 0), columns
    //      whose distance from the sinogram's center column (the rotation axis, assuming the
    //      sinogram was already shifted onto center) falls in [maskInnerRadius, maskOuterRadius)
    //      are restored to their pre-filter values, on both sides of center. A real object edge
    //      centered on the rotation axis produces an angle-invariant column band indistinguishable
    //      from a stripe artifact to this filter, so this excludes it from filtering instead of
    //      letting it get suppressed. maskInnerRadius = 0 protects a solid disk out to
    //      maskOuterRadius; a nonzero maskInnerRadius protects only the annulus between the two.
    static void remove_stripes(cv::Mat& sinogram, int level, double sigma, int order = 3, int pad = 200,
                                int maskInnerRadius = 0, int maskOuterRadius = 0);

    // Complementary large/wide-stripe removal (ports the "large stripe" detection idea from
    // Vo, Atwood & Drakopoulos, Opt. Express 26(22), 2018 - median-profile anomaly detection -
    // adapted to the additive log-attenuation domain our sinograms are already in, rather than
    // the multiplicative transmission-ratio domain that paper's reference implementation uses).
    //
    // remove_stripes targets stripes via their angular-frequency signature and is limited by how
    // deep the wavelet decomposition goes; this instead looks directly at each column's typical
    // value across all angles:
    //   1. medianProfile[col] = median over all angles of sinogram(:, col).
    //      A real stripe is ~constant across angle, so it shows up here; real sample content
    //      mostly averages out across a full angular range.
    //   2. smoothProfile = medianProfile smoothed with a wide (smoothWindow) median filter - the
    //      "should be" slowly-varying background trend, with any stripe's localized deviation
    //      averaged away by the wide window.
    //   3. deviation[col] = medianProfile[col] - smoothProfile[col].
    //   4. Columns where |deviation| exceeds snr * MAD(deviation) (a robust noise estimate) are
    //      flagged as stripe columns, then dilated by one column each side to catch their edges.
    //   5. For each flagged column, deviation[col] is subtracted from every row of that column -
    //      removing the angle-invariant bias while leaving each row's real content untouched.
    //
    // IMPORTANT: because step 2 uses a MEDIAN filter, smoothWindow must be more than roughly
    // 2x the widest stripe you're targeting. If a stripe occupies more than half of most window
    // positions that overlap it, the median gets captured by the stripe's own value instead of
    // the true background, deviation collapses to ~0, and the stripe silently passes through
    // uncorrected - verified experimentally: a 40px-wide injected stripe was completely missed
    // at smoothWindow=51, but >99% corrected at smoothWindow=101. When in doubt, err wide - an
    // oversized window just costs a little precision at the stripe's edges, not a missed stripe.
    // maskInnerRadius/maskOuterRadius: same annulus-protection semantics as remove_stripes.
    static void remove_large_stripes(cv::Mat& sinogram, double snr, int smoothWindow,
                                      int maskInnerRadius = 0, int maskOuterRadius = 0);

    // Complementary small/thin-stripe removal (the "sorting technique", Vo, Atwood & Drakopoulos,
    // Opt. Express 26(22), 2018, Algorithm 3) - targets the opposite end of the spectrum from
    // remove_large_stripes: a single (or a few) detector columns with a slightly different
    // gain/offset than their neighbors, producing a thin ring that's nearly invisible to both
    // remove_stripes (too fine-grained for the wavelet decomposition to separate from real
    // texture) and remove_large_stripes (whose wide smoothing window is specifically unsuited to
    // narrow defects - see its docs).
    //
    // Works by exploiting the fact that a detector defect is angle-independent while real sample
    // content varies smoothly with angle:
    //   1. Sort each column independently along the angle axis. A real column's sorted profile
    //      varies smoothly rank-to-rank; a defective column's sorted profile is consistently
    //      offset from its neighbors at every rank, since the defect biases every angle the same way.
    //   2. At each rank, median-filter across columns with a small window - this pulls a
    //      consistently-offset column back toward its neighbors' value at that rank, while barely
    //      touching a column that already agrees with its neighbors there.
    //   3. Un-sort back to the original angle order using the sort permutation from step 1.
    //
    // window should be small - roughly the width of the defect in columns (a handful of pixels),
    // the opposite of remove_large_stripes' smoothWindow. Too large starts blurring real
    // structure across neighboring columns instead of just correcting the defective one.
    // maskInnerRadius/maskOuterRadius: same annulus-protection semantics as remove_stripes.
    static void remove_small_stripes(cv::Mat& sinogram, int window,
                                      int maskInnerRadius = 0, int maskOuterRadius = 0);
};

#endif // RING_FILTER_H
