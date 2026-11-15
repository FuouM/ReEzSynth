"""WebGPU image ops: the ``ezsynth_wgpu`` Rust module.

Kernels: ``rust/ebsynth-gpu/src/imageops.rs`` and ``shaders/imageops_*.wgsl``.
Warping and occlusion masks match OpenCV; histogram blending converts to Lab in float.
"""

from __future__ import annotations

import numpy as np

from ...utils.wgpu_runtime import import_wgpu

_ew = import_wgpu()

OPS = frozenset({"warp", "forward_splat", "pull_push", "hist_blend", "occlusion_masks"})


def warp(img: np.ndarray, flow: np.ndarray, *, nearest: bool) -> np.ndarray:
    """Backward-warp ``img`` by ``flow`` with reflected borders."""
    return _ew.warp(
        np.ascontiguousarray(img),
        np.ascontiguousarray(flow),
        nearest=nearest,
        border="reflect",
    )


def pull_push(
    color: np.ndarray, weight: np.ndarray, *, levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical hole filling; returns new ``(color, weight)`` arrays."""
    return _ew.pull_push(color, weight, levels=levels)


def forward_splat(
    img: np.ndarray,
    flow: np.ndarray,
    *,
    fill_holes: bool,
    src_guide: np.ndarray | None,
    tgt_guide: np.ndarray | None,
    uint8_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-splat float32 ``img`` along ``flow``; returns ``(image, raw weight)``.

    Guides (float32, both or neither) make the splat weights bilateral.
    """
    guides = {}
    if src_guide is not None and tgt_guide is not None:
        guides = {"src_guide": src_guide, "tgt_guide": tgt_guide}
    splat, raw_weight = _ew.forward_splat(img, flow, fill_holes=fill_holes, **guides)
    out = splat.clip(0, 255).astype(np.uint8) if uint8_out else splat
    return out, raw_weight


def hist_blend(
    a: np.ndarray,
    b: np.ndarray,
    error_mask: np.ndarray,
    weight1: float,
    weight2: float,
) -> np.ndarray:
    """``opencv.hist_blend`` on the GPU; ``error_mask`` must be (H, W)."""
    if error_mask.ndim != 2:
        raise ValueError("The wgpu histogram blend takes a (H, W) error mask.")
    return _ew.hist_blend(
        np.ascontiguousarray(a, dtype=np.uint8),
        np.ascontiguousarray(b, dtype=np.uint8),
        np.ascontiguousarray(error_mask, dtype=np.uint8),
        float(weight1),
        float(weight2),
    )


def occlusion_masks(
    fwd: np.ndarray,
    bwd: np.ndarray,
    *,
    alpha: float,
    beta: float,
    dilate: int,
    coverage: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Unreliable-pixel masks (uint8 0/255) of one forward/backward flow pair."""
    return _ew.occlusion_masks(
        np.ascontiguousarray(fwd, dtype=np.float32),
        np.ascontiguousarray(bwd, dtype=np.float32),
        alpha=alpha,
        beta=beta,
        dilate=dilate,
        coverage=coverage,
    )
