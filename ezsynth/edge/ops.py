"""Shared utilities for edge extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PstEdgeParams:
    """Defaults for Phycv PST."""

    S: float = 0.3
    W: int = 15
    sigma_LPF: float = 0.15
    thresh_min: float = 0.05
    thresh_max: float = 0.9
    morph_flag: int = 1


@dataclass(frozen=True)
class PageEdgeParams:
    """Defaults for Phycv PAGE."""

    mu_1: float = 0.0
    mu_2: float = 0.35
    sigma_1: float = 0.05
    sigma_2: float = 0.8
    S1: float = 0.8
    S2: float = 0.8
    sigma_LPF: float = 0.1
    thresh_min: float = 0.0
    thresh_max: float = 0.9
    morph_flag: int = 1


def merge_pst_params(
    overrides: PstEdgeParams | Mapping[str, Any] | None,
) -> PstEdgeParams:
    if overrides is None:
        return PstEdgeParams()
    if isinstance(overrides, PstEdgeParams):
        return overrides
    merged = asdict(PstEdgeParams())
    merged.update(overrides)
    return PstEdgeParams(**merged)


def merge_page_params(
    overrides: PageEdgeParams | Mapping[str, Any] | None,
) -> PageEdgeParams:
    if overrides is None:
        return PageEdgeParams()
    if isinstance(overrides, PageEdgeParams):
        return overrides
    merged = asdict(PageEdgeParams())
    merged.update(overrides)
    return PageEdgeParams(**merged)


PAD_SIZE = 16


def replace_zeros_tensor(image: torch.Tensor, replace_value: int = 1) -> torch.Tensor:
    zero_mask = image == 0
    replace_tensor = torch.full_like(image, replace_value)
    return torch.where(zero_mask, replace_tensor, image)


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def pad_gray_reflect(gray: np.ndarray, pad: int = PAD_SIZE) -> np.ndarray:
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)


def unpad_gray(gray: np.ndarray, pad: int = PAD_SIZE) -> np.ndarray:
    return gray[pad:-pad, pad:-pad]


def postprocess_pst_page(edge_map: np.ndarray) -> np.ndarray:
    edge_map = cv2.GaussianBlur(edge_map, (5, 5), 3)
    edge_map = edge_map * 255
    return edge_map.astype(np.uint8)


def to_bgr_three_channel(edge_map: np.ndarray) -> np.ndarray:
    """Ebsynth expects 3-channel BGR guides."""
    if edge_map.ndim == 2:
        return np.stack([edge_map] * 3, axis=-1)
    return edge_map
