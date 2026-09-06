#ifndef RING_REMOVAL_POLAR_H
#define RING_REMOVAL_POLAR_H
#include <opencv2/opencv.hpp>

// Post-reconstruction ring-artifact removal, operating on a reconstructed slice
// (as opposed to RingFilter in ring_filter.h/.cpp, which works on sinograms
// before reconstruction). Reimplements the algorithm behind tomopy's
// misc.corr.remove_ring / libtomo/misc/remove_ring.c (BSD-3-Clause, Argonne
// National Laboratory) - the "Ketcham-style" polar-coordinate ring remover.
//
// IMPORTANT PROVENANCE NOTE: this is built from a detailed technical
// description of that C source (obtained and paraphrased under copyright
// constraints that cap how much of it may be quoted), not a line-by-line port
// of the literal code. The description was cross-checked for internal
// consistency and the algorithm below is believed faithful to the original's
// intent and parameter semantics, but exact numerical parity with tomopy's
// output isn't guaranteed the way a byte-for-bit port would be. If exact
// parity matters, spot-check a slice against tomopy's own output.
//
// Algorithm, per slice:
//  1. Convert to polar coordinates centered on the rotation axis (nearest-
//     neighbor sampling; radius -> column, angle -> row). A ring (concentric
//     circle in Cartesian) becomes a value that's ~constant across every row
//     at one column in polar space. Samples are clamped to [threshMin,
//     threshMax] to bound the dynamic range before filtering.
//  2. Radial median filter: for each angle (row), median-filter across
//     radius (columns). Kernel radius grows in three radius bands (finer
//     near the center, where real structure changes fast with radius;
//     coarser further out) derived from ringWidth.
//  3. diff = polar - median_filtered, then zero any |diff| > thresh: a
//     genuine ring survives as a small, angle-persistent bias; a real sharp
//     radial edge in the sample produces a large diff and is rejected here.
//  4. Azimuthal mean filter: for each radius (column), mean-filter the
//     (thresholded) diff across angle (rows), circularly (wrapBoundary) or
//     mirrored at the boundary. Content that's genuinely angle-invariant
//     (a real ring) survives averaging with a large mean; noise or real
//     structure that varies with angle averages toward zero. A pixel whose
//     diff was zeroed in step 3 stays zero here rather than picking up a
//     neighbor-derived estimate. Kernel radius again grows in the same three
//     radius bands, derived from thetaMinDeg.
//  5. Inverse-transform the filtered polar buffer (now an estimate of just
//     the ring pattern) back to Cartesian and subtract it from the original
//     slice.
class PolarRingRemoval
{
public:
    // slice: CV_32FC1, square (n_detectors x n_detectors) reconstructed image, centered on
    // ((cols-1)/2, (rows-1)/2) - this app's FBP always centers the rotation axis there, so unlike
    // tomopy's Python API there's no separate center_x/center_y to pass in.
    // thresh: diff-from-median magnitude above which a pixel is rejected as "not ring-like".
    // threshMax/threshMin: clamp applied to samples during the polar-coordinate resampling step.
    // thetaMinDeg (0-179): only bias that persists across roughly this many degrees of angle (or
    // more) survives the azimuthal mean filter as a ring; narrower angular features average out.
    // ringWidth: sets the radial median filter's kernel size (as 2*ringWidth+1).
    // wrapBoundary: true = circular (WRAP) boundary for the azimuthal mean filter, false = mirror
    // (REFLECT).
    // parallel: true (default) parallelizes this call's internal loops across all cores - correct
    // for a single standalone call, e.g. the interactive B/M/T preview. Pass false when the caller
    // is already processing multiple slices concurrently from its own thread pool (as
    // post_process_worker.cpp does): nesting two independent thread pools oversubscribes the
    // machine (e.g. 12 outer x 12 inner threads fighting over 12 cores) and is slower than either
    // level of parallelism alone.
    // maskRadiusRatio (0-1, default 1.0 = whole square): the polar transform's radial extent is
    // capped at maskRadiusRatio * (min(rows,cols)/2) - pass the same circMaskRatio the
    // reconstruction's circular mask used (FbpReconstructor::reconstruct_slice) to skip processing
    // the annulus that mask already zeroed out. Since the polar buffer's size scales with radius^2,
    // this saves roughly (1 - maskRadiusRatio^2) of the work with NO change to the result: there's
    // no ring signal left to find in a region that's already been masked to zero.
    static cv::Mat remove_ring(const cv::Mat& slice, double thresh, double threshMax, double threshMin,
                                double thetaMinDeg, int ringWidth, bool wrapBoundary, bool parallel = true,
                                double maskRadiusRatio = 1.0);
};

#endif // RING_REMOVAL_POLAR_H
