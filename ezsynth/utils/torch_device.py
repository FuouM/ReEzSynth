"""Default PyTorch device: CUDA, else Apple MPS, else CPU."""

import torch


def get_auto_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"
