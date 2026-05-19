"""ComfyUI IMAGE tensor <-> ReEzSynth BGR uint8 conversions."""

from __future__ import annotations

import numpy as np
import torch


def _single_batch_image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] != 1:
        raise ValueError(
            f"ReEzSynth nodes expect batch size 1 (got {tensor.shape[0]}). "
            "Use a batch node or run one frame at a time."
        )
    return tensor[0]


def image_tensor_to_bgr_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI RGB float IMAGE to internal BGR uint8 HWC."""
    frame = _single_batch_image(tensor)
    rgb = frame.detach().cpu().numpy()
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb[..., :3] * 255.0).round().astype(np.uint8)[..., ::-1]
    return np.ascontiguousarray(bgr)


def bgr_uint8_to_image_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert internal BGR uint8 HWC to ComfyUI RGB float IMAGE [1,H,W,C]."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected BGR HWC image, got shape {image.shape}.")
    rgb = image[..., :3][..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)


def error_map_to_image_tensor(error_map: np.ndarray) -> torch.Tensor:
    """Normalize a float error map to a visible RGB IMAGE tensor."""
    err = np.asarray(error_map, dtype=np.float32)
    if err.ndim == 3 and err.shape[2] == 1:
        err = err[..., 0]
    max_err = float(err.max()) if err.size else 0.0
    if max_err > 1e-6:
        vis = (255.0 * (err / max_err)).round().astype(np.uint8)
    else:
        vis = np.zeros_like(err, dtype=np.uint8)
    if vis.ndim == 2:
        vis = np.stack([vis, vis, vis], axis=-1)
    elif vis.shape[2] == 1:
        vis = np.repeat(vis, 3, axis=2)
    else:
        vis = vis[..., :3]
    rgb = vis.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
