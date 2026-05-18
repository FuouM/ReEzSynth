"""Classic Gaussian-residual edge map."""

from __future__ import annotations

import cv2
import numpy as np

from .ops import create_gaussian_kernel


def compute_classic_edge(
    bgr: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 6.0,
) -> np.ndarray:
    kernel = create_gaussian_kernel(kernel_size, sigma)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.filter2D(gray, -1, kernel)
    edge_map = cv2.subtract(gray, blurred)
    edge_map = np.clip(edge_map + 128, 0, 255)
    return edge_map.astype(np.uint8)
