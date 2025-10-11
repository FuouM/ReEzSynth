# Backend Comparison

## ImageSynth

> Minimal config

| backend | hardware | segment | stylit | facestyle | total |
|-|-|-|-|-|-|
| CUDA | RTX3060 - Ryzen 5 3600 | 0.2707 s | 0.5899 s | 0.4699 s | 1.6146 s |
| CUDA PyTorch JIT | - | 0.2695 s | 0.6047 s | 0.4607 s | 1.5307 s |
| PyTorch CUDA (ncc) | - | 1.1733 s | 14.3003 s | 8.9392 s | 24.6881 s |
| PyTorch CUDA (ssd) | - | 1.1446 s | 14.0865 s | 8.7831 s | 24.2961 s |
| PyTorch CUDA (residual ncc) | - | 0.7209 s | 5.2828 s | 3.8109 s | 10.0852 s |
| PyTorch CUDA (residual ssd) | - | 0.7606 s | 4.9518 s | 3.5809 s | 9.5445 s |
| PyTorch CPU (ncc) | Apple M4 24GB | 4.9055 s | 82.2316 s | 51.5455 s | 138.7563 s |
| PyTorch MPS (ncc) | - | 3.3720 s | 333.4654 s | 140.8887 s | 478.8690 s |
| PyTorch CPU (residual ncc) | - | 2.2292 s | 27.5490 s | 20.1376 s | 49.9900 s |
| PyTorch MPS (residual ncc) | - | 1.3780 s | 31.8284 s | 38.7016 s | 73.4036 s |

> PyTorch residual matches CUDA quality and faster than Non-residual
>
> PyTorch Non-residual looks better than CUDA for low params
>
> SSD uses a bit less memory than NCC
