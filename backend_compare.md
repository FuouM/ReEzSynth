# Backend Comparison

## ImageSynth

> Minimal config

| backend | hardware | segment | stylit | facestyle | total |
|-|-|-|-|-|-|
| CUDA | RTX3060 - Ryzen 5 3600 | 0.2707 s | 0.5899 s | 0.4699 s | 1.6146 s |
| PyTorch CUDA | - | 2.0054 s | 34.1567 s | 23.1453 s | 59.7566 s |
| PyTorch+Metal | Apple M4 24GB | 1.2921 s | 12.5120 s | 8.3418 s | 22.2256 s |
| PyTorch MPS | - | 1.6451 s | 15.4771 s | 11.1205 s | 28.2856 s |
