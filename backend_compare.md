# Backend Comparison

## ImageSynth

### Windows

> RTX3060 - Ryzen 5 3600. Minimal config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA PyTorch JIT (ssd) | 0.3062 s | 0.7650 s | 0.5482 s | 1.8252 s |
| CUDA PyTorch JIT (ncc) | 0.3098 s | 0.8086 s | 0.6349 s | 1.9764 s |
| PyTorch CUDA (ssd) | 0.7432 s | 5.0049 s | 3.7381 s | 9.8173 s |
| PyTorch CUDA (ncc) | 0.7956 s | 6.3167 s | 3.7569 s | 11.3063 s |

> RTX3060 - Ryzen 5 3600. Full config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| CUDA PyTorch JIT (ssd) | 0.3840 s | 1.0661 s | 0.7403 s | 2.3945 s |
| CUDA PyTorch JIT (ncc) | 0.5617 s | 3.2818 s | 1.1257 s | 5.1724 s |
| PyTorch CUDA (ssd - memclr) | 7.1243 s | 43.9366 s | 33.0244 s | 84.4305 s |
| PyTorch CUDA (ncc - memclr) | 7.4011 s | 74.2658 s | 35.0005 s | 116.9856 s |

Notes:

- SSD for results aligning with the original implementation.
- NCC takes longer but are more robust to changes.
- More guides -> More memory usage.

### MacOS (Apple Silicon)

> M4 24GB. Minimal config. Residual Transfer: True

| backend | segment | stylit | facestyle | total |
|-|-|-|-|-|
| PyTorch CPU (ssd) | - s | - s | - s | - s |
| PyTorch CPU (ncc) | - s | - s | - s | - s |
| PyTorch MPS (ssd) | - s | - s | - s | - s |
| PyTorch MPS (ncc) | - s | - s | - s | - s |
