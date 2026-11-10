"""ComfyUI IMAGE tensors <-> ReEzSynth BGR uint8 arrays.

ComfyUI ``IMAGE`` is a float32 tensor ``[B, H, W, C]`` (RGB, values in ``[0, 1]``,
usually on the CPU). ReEzSynth works on BGR uint8 ``HWC`` NumPy arrays. Conversions
are whole-batch tensor ops: one quantization pass and one host copy per call.
"""

from __future__ import annotations

import numpy as np
import torch

_BGR = [2, 1, 0]


def _check_image_batch(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
        raise ValueError(f"Expected an IMAGE tensor [B,H,W,C], got shape {shape}.")
    if tensor.shape[0] < 1:
        raise ValueError("Expected at least one IMAGE frame.")
    if tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected 1, 3 or 4 channels in IMAGE tensor, got {tensor.shape[-1]}."
        )


def image_batch_to_bgr_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Quantize an IMAGE batch to a contiguous BGR uint8 array ``[B, H, W, 3]``.

    Alpha is dropped; single-channel images are broadcast to gray BGR.
    """
    _check_image_batch(tensor)
    rgb = tensor.detach()
    if rgb.shape[-1] == 1:
        rgb = rgb.expand(*rgb.shape[:-1], 3)
    bgr = rgb[..., _BGR]  # advanced indexing: one copy, also drops alpha
    bgr = bgr.float().clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    return bgr.cpu().numpy()


def image_batch_to_bgr_uint8_list(tensor: torch.Tensor) -> list[np.ndarray]:
    """IMAGE batch -> list of BGR uint8 ``HWC`` frames (views into one array)."""
    return list(image_batch_to_bgr_uint8(tensor))


def image_tensor_to_bgr_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Single-image IMAGE ``[1, H, W, C]`` -> BGR uint8 ``HWC``."""
    _check_image_batch(tensor)
    if tensor.shape[0] != 1:
        raise ValueError(
            f"Expected a single image (batch size 1), got a batch of {tensor.shape[0]}. "
            "Select one image with 'Get Image from Batch'."
        )
    return image_batch_to_bgr_uint8(tensor)[0]


def bgr_uint8_sequence_to_image_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """BGR uint8 ``HWC`` frames (same size) -> IMAGE float32 ``[B, H, W, 3]``."""
    if not frames:
        raise ValueError("Expected at least one frame.")
    shape = frames[0].shape
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected BGR HWC frames, got shape {frame.shape}.")
        if frame.shape != shape:
            raise ValueError(
                f"All frames must share one shape, got {shape} and {frame.shape}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frames, got {frame.dtype}.")
    batch = torch.from_numpy(np.stack(frames, axis=0))
    return batch[..., _BGR].float().div_(255.0)


def bgr_uint8_to_image_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR uint8 ``HWC`` -> IMAGE float32 ``[1, H, W, 3]``."""
    return bgr_uint8_sequence_to_image_tensor([image])


def error_map_to_image_tensor(error_map: np.ndarray) -> torch.Tensor:
    """Float error map ``HW`` / ``HW1`` / ``HWC`` -> gray IMAGE normalized to its max."""
    err = torch.as_tensor(np.asarray(error_map), dtype=torch.float32)
    if err.ndim == 3:
        err = err[..., 0] if err.shape[2] == 1 else err[..., :3].mean(dim=2)
    if err.ndim != 2:
        raise ValueError(f"Expected a 2D error map, got shape {tuple(err.shape)}.")
    max_err = float(err.max()) if err.numel() else 0.0
    vis = err / max_err if max_err > 1e-6 else torch.zeros_like(err)
    return (
        vis.clamp_(0.0, 1.0).unsqueeze(-1).expand(-1, -1, 3).unsqueeze(0).contiguous()
    )
