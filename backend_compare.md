# Backend Comparison

<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD045 -->
<!-- markdownlint-disable MD060 -->

## ImageSynth

### Windows

> RTX3060 - Ryzen 5 3600. Minimal config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT  | 0.3519 s  |  0.8217 s | 0.6548 s |  2.0702 s  |
| CPU C++  PyTorch JIT  | 0.8753 s  |  9.5893 s | 6.4620 s | 17.1807 s  |
| PyTorch CUDA          | 0.7432 s  |  5.0049 s | 3.7381 s |  9.8173 s  |

> RTX3060 - Ryzen 5 3600. Full config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT  | 0.3840 s |  1.0661 s |  0.7403 s |  2.3945 s |
| CPU C++  PyTorch JIT  | 3.1943 s | 32.5626 s | 20.5413 s | 56.5844 s |
| PyTorch CUDA          | 7.1243 s | 43.9366 s | 33.0244 s | 84.4305 s |

### MacOS M4

> Minimal config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CPU C++ PyTorch JIT | 0.2178 s  |  2.3356 s |  1.6329 s |  4.2787 s |
| PyTorch MPS         | 1.8573 s  | 39.0842 s | 24.4787 s | 65.5494 s |

> Full config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CPU C++ PyTorch JIT |  0.7631 s  |   7.8879 s |   5.0888 s |  13.8319 s  |
| PyTorch MPS         | 16.9250 s  | 296.8260 s | 178.6254 s | 492.7071 s  |

Notes:

- SSD for results aligning with the original implementation.
- NCC takes longer but are more robust to changes.
- More guides -> More memory usage.
