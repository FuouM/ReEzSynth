"""Flow consistency masks and conservative fill helpers."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _grid(height: int, width: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    return np.stack((x, y), axis=-1).astype(np.float32)


def _as_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32)


def _sample_flow(flow: np.ndarray, coords: np.ndarray) -> np.ndarray:
    coords_f32 = _as_float32(coords)
    return cv2.remap(
        _as_float32(flow),
        coords_f32[..., 0],
        coords_f32[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _outside(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        (coords[..., 0] < 0)
        | (coords[..., 0] > width - 1)
        | (coords[..., 1] < 0)
        | (coords[..., 1] > height - 1)
    )


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
    grid = _grid(h, w)
    target_coords = grid + primary_flow
    sampled_secondary = _sample_flow(secondary_flow, target_coords)

    residual_sq = np.sum((primary_flow + sampled_secondary) ** 2, axis=2)
    flow_mag_sq = np.sum(primary_flow**2, axis=2) + np.sum(sampled_secondary**2, axis=2)
    threshold = alpha * flow_mag_sq + beta

    mask = (residual_sq > threshold) | _outside(target_coords, h, w)
    out = mask.astype(np.uint8) * 255
    if dilate > 0:
        k = 2 * dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel)
    return out


def _coverage_holes(flow: np.ndarray) -> np.ndarray:
    """Approximate target pixels not reached by rounded forward splats."""
    h, w = flow.shape[:2]
    coords = _grid(h, w) + _as_float32(flow)
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


def compute_flow_occlusion_masks(
    fwd_flows: List[np.ndarray],
    bwd_flows: List[np.ndarray],
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
    dilate: int = 1,
    coverage: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build per-transition unreliable masks for both temporal directions.

    ``fwd_masks[i]`` marks unreliable pixels in frame ``i + 1`` when traveling
    from frame ``i`` to ``i + 1``. ``bwd_masks[i]`` marks unreliable pixels in
    frame ``i`` when traveling backward from frame ``i + 1`` to ``i``.
    """
    if len(fwd_flows) != len(bwd_flows):
        raise ValueError(
            f"Forward/backward flow length mismatch: {len(fwd_flows)} vs {len(bwd_flows)}"
        )

    fwd_masks = []
    bwd_masks = []
    for fwd, bwd in zip(fwd_flows, bwd_flows):
        fwd_mask = _consistency_mask(
            bwd,
            fwd,
            alpha=alpha,
            beta=beta,
            dilate=dilate,
        )
        bwd_mask = _consistency_mask(
            fwd,
            bwd,
            alpha=alpha,
            beta=beta,
            dilate=dilate,
        )
        if coverage:
            fwd_mask = np.maximum(fwd_mask, _coverage_holes(fwd).astype(np.uint8) * 255)
            bwd_mask = np.maximum(bwd_mask, _coverage_holes(bwd).astype(np.uint8) * 255)
        fwd_masks.append(fwd_mask)
        bwd_masks.append(bwd_mask)
    return fwd_masks, bwd_masks


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return mask
    return np.repeat(mask[..., None], 3, axis=2)


def modulation_from_mask(mask: np.ndarray, floor: int = 64) -> np.ndarray:
    """Create a guide modulation map that only relaxes unreliable masked regions."""
    floor = int(np.clip(floor, 0, 255))
    mod = np.where(mask > 0, floor, 255).astype(np.uint8)
    return mask_to_bgr(mod)


def inpaint_masked_regions(
    base: np.ndarray,
    mask: np.ndarray,
    *,
    radius: int = 3,
    feather: int = 5,
) -> np.ndarray:
    """Inpaint masked pixels to avoid warped-border artifacts."""
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return base

    filled = cv2.inpaint(base, binary, radius, cv2.INPAINT_TELEA)
    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        alpha = cv2.GaussianBlur(binary, (k, k), 0).astype(np.float32) / 255.0
    else:
        alpha = binary.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    out = base.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def composite_masked_regions(
    base: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    *,
    feather: int = 5,
) -> np.ndarray:
    """Composite overlay into base only where mask is non-zero."""
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return base

    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        alpha = cv2.GaussianBlur(binary, (k, k), 0).astype(np.float32) / 255.0
    else:
        alpha = binary.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    out = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)
