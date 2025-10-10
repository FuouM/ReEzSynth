# Backend Comparison

## ImageSynth

> Minimal config

| backend | hardware | segment | stylit | facestyle | total |
|-|-|-|-|-|-|
| CUDA | RTX3060 - Ryzen 5 3600 | 0.2707 s | 0.5899 s | 0.4699 s | 1.6146 s |
| PyTorch CUDA | - | 1.1733 s | 14.3003 s | 8.9392 s | 24.6881 s |
| PyTorch CUDA (residual) | - | 0.7323 s | 6.1073 s | 3.8444 s | 11.1140 s |
| PyTorch+Metal (old) | Apple M4 24GB | 1.2921 s | 12.5120 s | 8.3418 s | 22.2256 s |
| PyTorch MPS (old) | - | 1.6451 s | 15.4771 s | 11.1205 s | 28.2856 s |

> PyTorch residual matches CUDA quality and faster than Non-residual
>
> PyTorch Non-residual looks better than CUDA for low params
