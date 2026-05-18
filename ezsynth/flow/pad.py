"""Padding for NeuFlow spatial alignment."""

from __future__ import annotations

import numpy as np

# NeuFlow merges s8 and upsampled s16 feature maps; H,W must satisfy
# h//8 == 2*(h//16), etc.
NEUFLOW_SPATIAL_ALIGN = 32


def pad_bgr_to_stride(
    img: np.ndarray,
    stride: int = NEUFLOW_SPATIAL_ALIGN,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad bottom/right with edge values so H and W are multiples of ``stride``."""
    h, w, _ = img.shape
    nh = ((h + stride - 1) // stride) * stride
    nw = ((w + stride - 1) // stride) * stride
    pb, pr = nh - h, nw - w
    if pb == 0 and pr == 0:
        return img, (h, w, 0, 0)
    padded = np.pad(img, ((0, pb), (0, pr), (0, 0)), mode="edge")
    return padded, (h, w, pb, pr)
