#ifndef RING_REMOVAL_POLAR_CUDA_H
#define RING_REMOVAL_POLAR_CUDA_H
#include <opencv2/opencv.hpp>
#include <memory>

// GPU-backed implementation of PolarRingRemoval::remove_ring (see ring_removal_polar.h for the
// algorithm description and provenance note - this follows the exact same formulas and stage
// order, just as CUDA kernels instead of CPU loops, so results should agree closely with the CPU
// path, within float rounding and one deliberate algorithmic difference: the radial median filter
// finds the median via an O(k^2) per-pixel rank count instead of std::nth_element, since CUDA
// device code can't easily do an in-place partial sort per thread. That's fine for the ring widths
// this is actually used with (tens of elements); it gets quadratically slower for extreme settings
// (very large ring width), unlike the CPU path's near-linear cost.
//
// One instance is meant to be constructed once per reconstruction run (sized by the slice's
// rows/cols, which are constant for that whole run) and reused across every slice - mirroring
// FbpReconstructor/FbpCudaBackend's design, including the reason why: this class's device buffers
// are shared, mutable, per-instance state, and ReconstructionWorker::run() calls it from its
// multi-threaded per-row pool exactly like it does FbpReconstructor. Every call is internally
// serialized with a mutex (the GPU is one device anyway) so that's safe.
class PolarRingCudaBackend
{
public:
    // rows, cols: the reconstructed slice's fixed dimensions for this run.
    // maskRadiusRatio (0-1, default 1.0): caps the polar transform's radial extent at
    // maskRadiusRatio * (min(rows,cols)/2) - pass the same circMaskRatio the reconstruction's
    // circular mask used, to skip the annulus that mask already zeroed out (no ring signal left to
    // find there). Fixed at construction, like rows/cols, since it determines the cached buffer
    // sizes - see ring_removal_polar.h's remove_ring for the full rationale.
    PolarRingCudaBackend(int rows, int cols, double maskRadiusRatio = 1.0);
    ~PolarRingCudaBackend();

    PolarRingCudaBackend(const PolarRingCudaBackend&) = delete;
    PolarRingCudaBackend& operator=(const PolarRingCudaBackend&) = delete;

    // Same parameters/semantics as PolarRingRemoval::remove_ring.
    cv::Mat remove_ring(const cv::Mat& slice, double thresh, double threshMax, double threshMin,
                         double thetaMinDeg, int ringWidth, bool wrapBoundary);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // RING_REMOVAL_POLAR_CUDA_H
