"""Taichi image ops (kernels in ``taichi_kernels``); importing initializes Taichi."""

from __future__ import annotations

import numpy as np

from ...utils.taichi_runtime import ensure_ti_init
from . import taichi_kernels as K

ensure_ti_init()

OPS = frozenset({"warp", "forward_splat", "pull_push", "hist_blend", "poisson_cg"})


def warp(img: np.ndarray, flow: np.ndarray, *, nearest: bool) -> np.ndarray:
    """Bilinear backward warp with reflected borders."""
    if nearest:
        raise ValueError("The taichi image ops engine warps bilinearly only.")
    dst = np.zeros_like(img)
    K.bilinear_warp_kernel(img, flow, dst)
    return dst


def pull_push(
    color: np.ndarray, weight: np.ndarray, *, levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical hole filling; fills ``color`` / ``weight`` in place."""
    pyramid_color = [color]
    pyramid_weight = [weight]

    for _ in range(levels - 1):
        h, w = pyramid_color[-1].shape[:2]
        if h <= 2 or w <= 2:
            break
        h_next, w_next = h // 2, w // 2
        color_shape = pyramid_color[-1].shape[2:]
        next_color = np.zeros((h_next, w_next, *color_shape), dtype=np.float32)
        next_weight = np.zeros((h_next, w_next), dtype=np.float32)
        K.pull_kernel(
            pyramid_color[-1],
            pyramid_weight[-1],
            next_color,
            next_weight,
        )
        pyramid_color.append(next_color)
        pyramid_weight.append(next_weight)

    for i in range(len(pyramid_color) - 2, -1, -1):
        K.push_kernel(
            pyramid_color[i + 1],
            pyramid_weight[i + 1],
            pyramid_color[i],
            pyramid_weight[i],
        )
    return color, weight


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
    h, w = flow.shape[:2]
    use_bilateral = src_guide is not None and tgt_guide is not None
    dst_color = np.zeros_like(img)
    dst_weight = np.zeros((h, w), dtype=np.float32)
    empty_guide = np.zeros((1, 1, 3), dtype=np.float32)

    K.soft_splat_kernel(
        img,
        flow,
        dst_color,
        dst_weight,
        src_guide if use_bilateral else empty_guide,
        tgt_guide if use_bilateral else empty_guide,
        use_bilateral,
    )

    raw_weight = dst_weight.copy()
    if fill_holes:
        dst_color, dst_weight = pull_push(dst_color, dst_weight, levels=5)

    out = np.zeros_like(img, dtype=np.uint8 if uint8_out else np.float32)
    K.normalize_splat_kernel(dst_color, dst_weight, out, uint8_out)
    return out, raw_weight


def hist_blend(
    a: np.ndarray,
    b: np.ndarray,
    error_mask: np.ndarray,
    weight1: float,
    weight2: float,
) -> np.ndarray:
    """Approximate float-Lab variant of ``opencv.hist_blend``.

    The second pass takes the statistics of the blended float buffer on the CPU.
    """
    ma, sa = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    mb, sb = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    mme, sme = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    m_lab = (
        np.where(error_mask[..., None] == 0, a, b)
        if len(error_mask.shape) == 2
        else np.where(error_mask == 0, a, b)
    )

    K.compute_stats_kernel(a, ma, sa)
    K.compute_stats_kernel(b, mb, sb)
    K.compute_stats_kernel(m_lab, mme, sme)

    temp_lab = np.zeros_like(a, dtype=np.float32)
    K.hist_blend_apply_kernel(a, b, ma, sa, mb, sb, weight1, weight2, temp_lab)

    out = np.zeros_like(a)
    mab = np.mean(temp_lab, axis=(0, 1))
    sab = np.std(temp_lab, axis=(0, 1))
    K.hist_blend_final_pass_kernel(temp_lab, mab, sab, mme, sme, out)
    return out


def poisson_cg(
    gx: np.ndarray,
    gy: np.ndarray,
    target: np.ndarray,
    weight: float,
    *,
    max_iter: int,
) -> np.ndarray:
    """Conjugate-gradient screened Poisson solve of one channel (H, W)."""
    return K.poisson_solver_cg(gx, gy, target, weight, max_iter=max_iter)
