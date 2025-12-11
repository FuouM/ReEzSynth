from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .image_utils import _to_uint8

__all__ = [
    "_bgr_to_yuv_torch",
    "_fill_convex_poly_torch",
    "_draw_ellipse_torch",
    "get_skin_mask",
    "alpha_blend",
]


def _bgr_to_yuv_torch(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to YUV using PyTorch (matches cv2.cvtColor)."""
    # BGR to RGB first
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0

    # RGB to YUV conversion matrix (ITU-R BT.601)
    # Y = 0.299*R + 0.587*G + 0.114*B
    # U = -0.14713*R - 0.28886*G + 0.436*B + 128
    # V = 0.615*R - 0.51499*G - 0.10001*B + 128
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5

    yuv = np.stack([y, u, v], axis=-1)
    return (yuv * 255.0).astype(np.uint8)


def _fill_convex_poly_torch(
    mask: np.ndarray, points: np.ndarray, value: float
) -> np.ndarray:
    """Fill a convex polygon using rasterization."""
    from matplotlib.path import Path as MplPath

    h, w = mask.shape
    y_coords, x_coords = np.mgrid[:h, :w]
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    path = MplPath(points)
    inside = path.contains_points(coords).reshape(h, w)
    mask[inside] = value
    return mask


def _draw_ellipse_torch(
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


def get_skin_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Tuple[int, int]]
) -> np.ndarray:
    """Generate skin mask using PyTorch operations."""
    lm = np.asarray([(int(x), int(y)) for x, y in landmarks], dtype=np.int32)
    face_contour = lm[:17]
    face_width = face_contour[-1, 0] - face_contour[0, 0]
    forehead_roi = (
        face_contour[0, 0],
        max(face_contour[0, 1] - int(face_width * 0.75), 0),
        face_width,
        min(int(face_width * 0.75), face_contour[0, 1]),
    )
    x, y, w, h = forehead_roi
    forehead = image_bgr[y : y + h, x : x + w].copy()
    forehead_yuv = _bgr_to_yuv_torch(forehead)

    sample_points = [
        (int((w / 4) * 1), max(h - int(face_width / 4), 0)),
        (int((w / 4) * 2), max(h - int(face_width / 4), 0)),
        (int((w / 4) * 3), max(h - int(face_width / 4), 0)),
    ]
    samples = []
    for sx, sy in sample_points:
        sy = int(np.clip(sy, 0, h - 1))
        sx = int(np.clip(sx, 0, w - 1))
        samples.append(
            np.mean(
                forehead_yuv[max(0, sy - 5) : sy + 6, max(0, sx - 5) : sx + 6],
                axis=(0, 1),
            )
        )
    samples = np.array(samples)

    mask = np.zeros((h, w), dtype=np.float32)
    threshold = 50.0
    for row in range(h):
        for col in range(w):
            pix = forehead_yuv[row, col].astype(np.float32)
            errs = np.sum((samples - pix)[:, 1:] ** 2, axis=1)
            if np.min(errs) < threshold:
                mask[row, col] = 1.0

    full_mask = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    full_mask[y : y + h, x : x + w] = mask
    full_mask = _fill_convex_poly_torch(full_mask, face_contour, 1.0)
    center = tuple(
        (face_contour[0] + (face_contour[-1] - face_contour[0]) // 2).tolist()
    )
    axes = (int(face_width / 2), int(face_width / 2.5))
    full_mask = _draw_ellipse_torch(full_mask, center, axes)
    return full_mask


def alpha_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    sigma: float = 25.0,
) -> np.ndarray:
    """Alpha blend using PyTorch for blur operation."""
    fg = foreground.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    elif a.ndim == 3 and a.shape[2] == 1:
        pass
    elif a.ndim == 3 and a.shape[2] == 3:
        # already 3 channels
        pass
    else:
        a = a.reshape(alpha.shape[0], alpha.shape[1], -1)
        if a.shape[2] != 1 and a.shape[2] != 3:
            a = a[:, :, :1]

    # Use PyTorch for blur
    k = max(1, int(sigma))
    a_t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    a_t = F.avg_pool2d(a_t, kernel_size=k, stride=1, padding=k // 2)
    a = a_t.squeeze(0).permute(1, 2, 0).numpy()

    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = np.clip(a, 0.0, 1.0)
    out = fg * a + bg * (1.0 - a)
    return _to_uint8(out * 255.0)
