from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "clamp_landmarks",
    "get_head_area_rect",
]


def clamp_landmarks(
    landmarks: Sequence[Tuple[int, int]], size: Tuple[int, int]
) -> list[Tuple[int, int]]:
    h, w = size
    clamped = []
    for x, y in landmarks:
        clamped.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    return clamped


def get_head_area_rect(
    landmarks: Sequence[Tuple[int, int]], img_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    pts = np.asarray(landmarks, dtype=np.int32)
    width = pts[16, 0] - pts[0, 0]
    higher_y = min(pts[0, 1], pts[16, 1])
    height = pts[8, 1] - (higher_y - width // 2)
    x = max(int(pts[0, 0] - width * 0.1), 0)
    y = max(int((higher_y - width / 2.0) - (height * 0.2)), 0)
    max_w = img_size[1] - x
    max_h = img_size[0] - y
    w = int(min(width * 1.2, max_w))
    h = int(min(height * 1.4, max_h))
    return (x, y, w, h)
