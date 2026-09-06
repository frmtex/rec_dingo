// phase_retrieval.cpp
//
// C++/Accelerate port of a CuPy-based Paganin single-distance phase
// retrieval filter.
//
// Original algorithm (per-pixel, operating on a real transmission image):
//   fim   = FFT2(image)
//   filt  = 1 - (2*alpha/pix^2) * (cos(kx*pix) + cos(ky*pix) - 2)
//   phret = -log( |IFFT2(fim / filt)| )
//
// IMPORTANT CAVEATS BEFORE YOU RUN THIS ON REAL DATA:
//
// 1. vDSP_fft2d_zip requires both dimensions to be powers of two. CT
//    detector images almost never are. Two options:
//      a) Zero-pad (or edge-pad) nx/ny up to the next power of two,
//         run the filter, then crop back to the original size. This
//         is standard practice for Paganin-style filters anyway,
//         since it also suppresses edge-wraparound artifacts from the
//         periodic FFT boundary condition.
//      b) Use vDSP_DFT (mixed-radix, arbitrary composite lengths) with
//         separable row-then-column 1D transforms instead of
//         vDSP_fft2d_zip. More code, no padding required. Ask if you
//         want this version instead.
//    This file assumes (a): pad to power of two before calling.
//
// 2. Accelerate's 2D FFT stride/dimension-order convention (which
//    argument is "rows" vs "columns") is a known source of silent
//    transposition bugs. VERIFY against your CuPy reference on a small
//    test image (e.g. 8x8) before trusting this on real data -- compare
//    element-by-element, not just visually.
//
// 3. vDSP FFT calls are unnormalized in both directions. A forward+
//    inverse round trip multiplies the result by nx*ny, so we divide
//    by nx*ny once after the inverse transform (see `scale` below).
//    This matches CuPy/NumPy's convention of normalizing only the
//    inverse transform by N.
//
// 4. This uses the plain complex-to-complex FFT (vDSP_fft2d_zip), not
//    the real-packed variant (zrip), even though the input is real.
//    zrip is faster (exploits Hermitian symmetry) but its Nyquist/DC
//    packing convention is fiddly to get right in 2D. Once this
//    version is verified correct, it's a reasonable next optimization.

