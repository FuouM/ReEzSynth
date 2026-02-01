# Backend Comparison

## ImageSynth

### Windows

> RTX3060 - Ryzen 5 3600. Minimal config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT  | 0.3519 s  |  0.8217 s | 0.6548 s |  2.0702 s  |
| CPU C++  PyTorch JIT  | 0.8207 s  |  9.3658 s | 6.4947 s | 16.9986 s  |
| PyTorch CUDA          | 0.7432 s  |  5.0049 s | 3.7381 s |  9.8173 s  |

> RTX3060 - Ryzen 5 3600. Full config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT  | 0.3840 s |  1.0661 s |  0.7403 s |  2.3945 s |
| CPU C++  PyTorch JIT  | 3.1580 s | 31.6481 s | 18.9574 s | 54.0724 s |
| PyTorch CUDA          | 7.1243 s | 43.9366 s | 33.0244 s | 84.4305 s |

Notes:

- SSD for results aligning with the original implementation.
- NCC takes longer but are more robust to changes.
- More guides -> More memory usage.
