# rec_dingo

A Qt6 desktop application for reconstructing X-ray computed tomography (CT) scans using
filtered backprojection (FBP), with GPU-accelerated reconstruction on NVIDIA hardware via CUDA.

## Features

- Full pipeline: flat-field correction, phase retrieval, tilt/center-of-rotation correction,
  ring-artifact removal, filtered backprojection, and post-processing (beam hardening, 16-bit
  conversion).
- CUDA-accelerated FBP reconstruction (cuFFT ramp filter + a custom backprojection kernel).
- Two complementary ring-artifact filters:
  - a sinogram-domain wavelet-Fourier filter (with large/small-stripe variants), and
  - a post-reconstruction, polar-coordinate filter (reimplemented from a technical description of
    tomopy's `misc.corr.remove_ring`), also available on GPU.
- Multi-threaded CPU fallback for every stage - the app runs without a GPU, just slower.
- Live, interactive B/M/T slice preview for tuning reconstruction parameters before committing to
  a full run.
- Reconstruction settings are saved back into the dataset's `settings.ini` after each stage, so a
  sinogram or corrected-projection dataset can be reloaded later and re-reconstructed with
  different settings without starting over.

## Requirements

- Qt 6
- OpenCV 4
- FFTW3 (single-precision, `fftw3f`)
- NVIDIA CUDA Toolkit (for GPU acceleration; the app builds and runs without it disabled, but the
  `.pro` file currently expects it - see below)

On Ubuntu:

```bash
sudo apt install build-essential libopencv-dev libfftw3-dev nvidia-cuda-toolkit
```

## Building

```bash
qmake rec_dingo.pro
make -j$(nproc)
```

The `.pro` file targets an NVIDIA Pascal GPU (`sm_61`, e.g. Quadro P2000) by default - adjust
`CUDA_ARCH` in `rec_dingo.pro` to match your GPU's compute capability.

## License

LGPLv3 - see [LICENSE](LICENSE).
