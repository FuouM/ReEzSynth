# Backend Comparison

<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD045 -->
<!-- markdownlint-disable MD060 -->

## ImageSynth

### Windows

> RTX3060 - Ryzen 5 3600. Minimal config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT       | 0.3519 s  |  0.8217 s | 0.6548 s |  2.0702 s  |
| CPU  C++ PyTorch JIT       | 0.7318 s  |  7.6034 s | 5.0442 s | 13.5924 s  |
| CPU  C++ OPTS PyTorch JIT  | 0.7510 s  |  8.2012 s | 5.5895 s | 14.7557 s  |
| PyTorch CUDA               | 0.9603 s  |  6.5712 s | 4.6607 s | 12.5952 s  |
| Taichi  CUDA               | 3.1430 s  |  1.6317 s | 1.3700 s |  6.3765 s  |

> RTX3060 - Ryzen 5 3600. Full config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA C++ PyTorch JIT       | 0.3840 s |  1.0661 s |  0.7403 s |  2.3945 s |
| CPU  C++ PyTorch JIT       | 2.8844 s | 26.8920 s | 18.7526 s | 48.7577 s |
| CPU  C++ OPTS PyTorch JIT  | 1.8046 s | 19.9784 s | 15.2228 s | 37.2326 s |
| PyTorch CUDA               | 8.4203 s | 48.2450 s | 37.0145 s | 94.0713 s |
| Taichi  CUDA               | 5.8581 s |  6.0665 s |  5.1537 s | 17.3375 s |

### MacOS M4

> Minimal config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CPU C++ PyTorch JIT            | 0.2230 s  |  2.2521 s |  1.5188 s |  4.0867 s |
| CPU C++ OPT PyTorch JIT        | 0.2175 s  |  2.3316 s |  1.5395 s |  4.1791 s |
| PyTorch CPU                    | 1.5803 s  | 46.0797 s | 17.2497 s | 65.0448 s |
| PyTorch MPS                    | 1.9048 s  | 18.6886 s |  9.5341 s | 30.3013 s |
| Taichi  MPS                    | 0.3984 s  |  0.7467 s |  0.5148 s |  1.7712 s |

> Full config. Residual Transfer: True, SSD

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CPU C++ PyTorch JIT            |  0.7796 s  |   7.4995 s |   5.1122 s |  13.4845 s  |
| CPU C++ OPTS PyTorch JIT       |  0.4749 s  |   5.3502 s |   3.7535 s |   9.6710 s  |
| PyTorch CPU                    | 16.9250 s  | 216.7993 s | 142.7609 s | 376.1542 s  |
| PyTorch MPS                    | 38.6542 s  | 205.5949 s | 205.1384 s | 449.7303 s  |
| Taichi  MPS                    |  0.9665 s  |   3.0835 s |   2.2306 s |   6.3870 s  |

Notes:

- SSD for results aligning with the original implementation.
- NCC takes longer but are more robust to changes.
- More guides -> More memory usage.
