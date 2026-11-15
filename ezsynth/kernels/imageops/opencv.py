"""OpenCV / NumPy image ops: the default engine and the reference for the others."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

OPS = frozenset({"warp", "hist_blend", "occlusion_masks", "poisson_lsq"})


def flow_grid(height: int, width: int) -> np.ndarray:
    """Pixel coordinates (H, W, 2) as float32 ``(x, y)``."""
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    return np.stack((x, y), axis=-1).astype(np.float32)


@lru_cache(maxsize=8)
def _shared_grid(height: int, width: int) -> np.ndarray:
    grid = flow_grid(height, width)
    grid.flags.writeable = False
    return grid


def _as_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32)


def sample_flow(flow: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``flow`` at ``coords`` (H, W, 2), zero outside."""
    coords_f32 = _as_float32(coords)
    return cv2.remap(
        _as_float32(flow),
        coords_f32[..., 0],
        coords_f32[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def outside(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    """Mask of ``coords`` that fall outside a ``height`` x ``width`` frame."""
    return (
        (coords[..., 0] < 0)
        | (coords[..., 0] > width - 1)
        | (coords[..., 1] < 0)
        | (coords[..., 1] > height - 1)
    )


def warp(img: np.ndarray, flow: np.ndarray, *, nearest: bool) -> np.ndarray:
    """Backward-warp ``img`` by ``flow`` with reflected borders."""
    grid = _shared_grid(*flow.shape[:2])
    return cv2.remap(
        img,
        grid[..., 0] + flow[..., 0],
        grid[..., 1] + flow[..., 1],
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def hist_blend(
    a: np.ndarray,
    b: np.ndarray,
    error_mask: np.ndarray,
    weight1: float,
    weight2: float,
) -> np.ndarray:
    """Blend BGR ``a`` and ``b`` in 8-bit Lab with the lower-error frame's statistics."""
    if len(error_mask.shape) == 2:
        error_mask = np.repeat(error_mask[:, :, np.newaxis], 3, axis=2)
    a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2Lab)
    min_error_lab = np.where(error_mask == 0, a_lab, b_lab)
    a_mean, a_std = np.mean(a_lab, axis=(0, 1)), np.std(a_lab, axis=(0, 1))
    b_mean, b_std = np.mean(b_lab, axis=(0, 1)), np.std(b_lab, axis=(0, 1))
    min_error_mean, min_error_std = (
        np.mean(min_error_lab, axis=(0, 1)),
        np.std(min_error_lab, axis=(0, 1)),
    )
    t_mean = np.full(3, 0.5 * 256, dtype=np.float32)
    t_std = np.full(3, (1 / 36) * 256, dtype=np.float32)
    a_lab_norm = ((a_lab - a_mean) * t_std / (a_std + 1e-6) + t_mean).astype(np.float32)
    b_lab_norm = ((b_lab - b_mean) * t_std / (b_std + 1e-6) + t_mean).astype(np.float32)
    ab_lab = (a_lab_norm * weight1 + b_lab_norm * weight2 - 128) / 0.5 + 128
    ab_mean, ab_std = np.mean(ab_lab, axis=(0, 1)), np.std(ab_lab, axis=(0, 1))
    ab_lab_final = (ab_lab - ab_mean) * min_error_std / (ab_std + 1e-6) + min_error_mean
    ab_lab_final = np.clip(np.round(ab_lab_final), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ab_lab_final, cv2.COLOR_Lab2BGR)


def _consistency_mask(
    primary_flow: np.ndarray,
    secondary_flow: np.ndarray,
    *,
    alpha: float,
    beta: float,
    dilate: int,
) -> np.ndarray:
    """Return target-space unreliable mask for primary then secondary flow."""
    h, w = primary_flow.shape[:2]
    primary_flow = _as_float32(primary_flow)
    target_coords = flow_grid(h, w) + primary_flow
    sampled_secondary = sample_flow(secondary_flow, target_coords)

    residual_sq = np.sum((primary_flow + sampled_secondary) ** 2, axis=2)
    flow_mag_sq = np.sum(primary_flow**2, axis=2) + np.sum(sampled_secondary**2, axis=2)
    threshold = alpha * flow_mag_sq + beta

    mask = (residual_sq > threshold) | outside(target_coords, h, w)
    out = mask.astype(np.uint8) * 255
    if dilate > 0:
        k = 2 * dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel)
    return out


def _coverage_holes(flow: np.ndarray) -> np.ndarray:
    """Approximate target pixels not reached by rounded forward splats."""
    h, w = flow.shape[:2]
    coords = flow_grid(h, w) + _as_float32(flow)
    xi = np.rint(coords[..., 0]).astype(np.int32)
    yi = np.rint(coords[..., 1]).astype(np.int32)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    covered = np.zeros((h, w), dtype=np.uint8)
    covered[yi[valid], xi[valid]] = 1
    covered = cv2.dilate(
        covered,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return covered == 0


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
    fwd_mask = _consistency_mask(bwd, fwd, alpha=alpha, beta=beta, dilate=dilate)
    bwd_mask = _consistency_mask(fwd, bwd, alpha=alpha, beta=beta, dilate=dilate)
    if coverage:
        fwd_mask = np.maximum(fwd_mask, _coverage_holes(fwd).astype(np.uint8) * 255)
        bwd_mask = np.maximum(bwd_mask, _coverage_holes(bwd).astype(np.uint8) * 255)
    return fwd_mask, bwd_mask


def poisson_lsq(
    gx: np.ndarray,
    gy: np.ndarray,
    target: np.ndarray,
    weight: float,
    *,
    max_iter: int | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Least-squares solve of one channel's screened Poisson system, ``u`` (H, W).

    Rows: ``weight * (u[i] - u[i+W]) = weight * gx[i]`` for flat ``i < (H-1)W``,
    ``weight * (u[i] - u[i+1]) = weight * gy[i]`` for flat ``i < H(W-1)`` and
    ``u = target``. These are the flat-index operators of the original SciPy
    ``construct_A_cpu`` (the horizontal pairs run along the flattened image), so the
    solution matches SciPy's ``lsqr`` on that matrix; ``web/src/core/passBlend.ts``
    solves the same system. Conjugate gradients on the normal equations, matrix-free,
    from ``target``; stops at relative residual ``tol`` or after ``max_iter`` steps.
    """
    h, w = target.shape
    n = h * w
    nx, ny = (h - 1) * w, h * (w - 1)
    k2 = float(weight) ** 2
    gx_flat = np.asarray(gx, dtype=np.float64).reshape(n)
    gy_flat = np.asarray(gy, dtype=np.float64).reshape(n)
    c = np.asarray(target, dtype=np.float64).reshape(n)

    def normal(u: np.ndarray) -> np.ndarray:
        out = u.copy()
        d = k2 * (u[:nx] - u[w:])
        out[:nx] += d
        out[w:] -= d
        d = k2 * (u[:ny] - u[1 : ny + 1])
        out[:ny] += d
        out[1 : ny + 1] -= d
        return out

    rhs = c.copy()
    rhs[:nx] += k2 * gx_flat[:nx]
    rhs[w:] -= k2 * gx_flat[:nx]
    rhs[:ny] += k2 * gy_flat[:ny]
    rhs[1 : ny + 1] -= k2 * gy_flat[:ny]

    u = c.copy()
    r = rhs - normal(u)
    p = r.copy()
    rr = float(r @ r)
    stop = tol * tol * max(float(rhs @ rhs), 1e-30)
    it = 0
    while rr > stop and (max_iter is None or it < max_iter):
        q = normal(p)
        alpha = rr / float(p @ q)
        u += alpha * p
        r -= alpha * q
        rr_next = float(r @ r)
        p = r + (rr_next / rr) * p
        rr = rr_next
        it += 1
    return u.reshape(h, w)
