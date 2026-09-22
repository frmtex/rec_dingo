#ifndef PHASE_RETRIEVAL_CUDA_H
#define PHASE_RETRIEVAL_CUDA_H
#include <memory>
#include <vector>

// GPU-backed implementation of the free function phase_retrieval() in proj_correction.cpp (a
// single-distance Paganin phase-retrieval filter) - same algorithm, same filter formula, just
// cuFFT's 2D C2C transform instead of FFTW's. No OpenCV dependency (operates on plain float
// buffers, like the CPU function it mirrors), so this header stays free of both CUDA and OpenCV
// types for anyone including it from ordinary C++.
//
// One instance is meant to be constructed once per scan (sized by the padded power-of-two
// dimensions run_scan() already computes, fixed for the whole scan) and reused across every
// projection - mirrors FbpCudaBackend/PolarRingCudaBackend/SpotFilterCudaBackend's design,
// including the mutex for the same reason those have one.
class PhaseRetrievalCudaBackend
{
public:
    // nx, ny: padded dimensions (both must be powers of two, same requirement as the CPU function).
    PhaseRetrievalCudaBackend(int nx, int ny);
    ~PhaseRetrievalCudaBackend();

    PhaseRetrievalCudaBackend(const PhaseRetrievalCudaBackend&) = delete;
    PhaseRetrievalCudaBackend& operator=(const PhaseRetrievalCudaBackend&) = delete;

    // Same semantics as phase_retrieval(): image must have nx*ny elements, row-major (nx rows x ny
    // columns). The on-device filter is rebuilt only when alpha/pix change from the previous call
    // (they're constant for a whole scan in practice, so this is normally a one-time cost).
    std::vector<float> retrieve(const std::vector<float>& image, float alpha, float pix);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // PHASE_RETRIEVAL_CUDA_H
