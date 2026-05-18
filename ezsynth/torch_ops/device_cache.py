"""GPU allocator cache clearing shared by torch_ops and backends."""

from __future__ import annotations

from typing import Union

import torch

from ..consts import TORCH_CUDA_CLEAR_CACHE, TORCH_MPS_CLEAR_CACHE

DeviceLike = Union[str, torch.device, None]


def clear_torch_device_cache(device: DeviceLike) -> None:
    """Run ``torch.mps.empty_cache`` or ``torch.cuda.empty_cache`` when enabled."""
    if device is None:
        return
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "cpu":
        return
    if d.type == "mps":
        if TORCH_MPS_CLEAR_CACHE and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return
    if d.type == "cuda":
        if TORCH_CUDA_CLEAR_CACHE and torch.cuda.is_available():
            torch.cuda.empty_cache()
