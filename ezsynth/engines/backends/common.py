import gc
from typing import Union

import torch
import torch.nn.functional as F

DeviceLike = Union[str, torch.device]


def resample_tensor(
    tensor: torch.Tensor, new_h: int, new_w: int, mode: str = "bilinear"
) -> torch.Tensor:
    """Resize an HWC image tensor with torch.nn.functional.interpolate."""
    if tensor.shape[0] == new_h and tensor.shape[1] == new_w:
        return tensor

    is_uint8 = tensor.dtype == torch.uint8

    tensor_float = tensor.permute(2, 0, 1).unsqueeze(0).float()

    resampled_float = F.interpolate(
        tensor_float, size=(new_h, new_w), mode=mode, align_corners=False
    )

    resampled = resampled_float.squeeze(0).permute(1, 2, 0)

    if is_uint8:
        return resampled.clamp(0, 255).to(torch.uint8).contiguous()

    return resampled.contiguous()


def random_init_nnf(
    device: DeviceLike,
    target_h: int,
    target_w: int,
    source_h: int,
    source_w: int,
    patch_size: int,
) -> torch.Tensor:
    """Random NNF field (H, W, 2) int32 on ``device`` with valid patch centers."""
    r = patch_size // 2
    rand_x = torch.randint(
        r,
        source_w - r,
        (target_h, target_w, 1),
        device=device,
        dtype=torch.int32,
    )
    rand_y = torch.randint(
        r,
        source_h - r,
        (target_h, target_w, 1),
        device=device,
        dtype=torch.int32,
    )
    return torch.cat([rand_x, rand_y], dim=2).contiguous()


def get_auto_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
