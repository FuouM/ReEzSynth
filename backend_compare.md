# Backend Comparison

## ImageSynth

> Minimal config

| backend | hardware | segment | stylit | facestyle | total |
|-|-|-|-|-|-|
| CUDA | RTX3060 - Ryzen 5 3600 | 0.2707 s | 0.5899 s | 0.4699 s | 1.6146 s |
| PyTorch CUDA | - | 1.4125 s | 14.6887 s | 9.6738 s | 26.0454 s |
| PyTorch+Metal | Apple M4 24GB | 1.2921 s | 12.5120 s | 8.3418 s | 22.2256 s |
| PyTorch MPS | - | 1.6451 s | 15.4771 s | 11.1205 s | 28.2856 s |
