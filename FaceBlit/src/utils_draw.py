from typing import Tuple

import numpy as np


def fill_convex_poly(mask: np.ndarray, points: np.ndarray, value: float) -> np.ndarray:
    """Fill a convex polygon by half-plane intersection (no matplotlib)."""
    h, w = mask.shape
    y_coords, x_coords = np.mgrid[:h, :w].astype(np.float64)
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    all_ge = np.ones((h, w), dtype=bool)
    all_le = np.ones((h, w), dtype=bool)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cross = (bx - ax) * (y_coords - ay) - (by - ay) * (x_coords - ax)
        all_ge &= cross >= 0
        all_le &= cross <= 0
    inside = all_ge | all_le
    mask[inside] = value
    return mask


def draw_ellipse(
    mask: np.ndarray, center: Tuple[int, int], axes: Tuple[int, int]
) -> np.ndarray:
    """Draw a filled ellipse on the mask."""
    h, w = mask.shape
    y_coords, x_coords = np.mgrid[:h, :w]

    cx, cy = center
    a, b = axes

    # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
    ellipse_mask = ((x_coords - cx) ** 2 / (a**2) + (y_coords - cy) ** 2 / (b**2)) <= 1
    mask[ellipse_mask] = 1.0
    return mask
