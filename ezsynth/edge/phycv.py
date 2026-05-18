"""PST / PAGE edge detectors via Phycv."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from phycv import PAGE_GPU, PST_GPU

from .ops import (
    PAD_SIZE,
    PageEdgeParams,
    PstEdgeParams,
    merge_page_params,
    merge_pst_params,
    pad_gray_reflect,
    replace_zeros_tensor,
    unpad_gray,
)


@lru_cache(maxsize=8)
def _pst_gpu(device: str) -> PST_GPU:
    return PST_GPU(device=device)


@lru_cache(maxsize=8)
def _page_gpu(device: str) -> PAGE_GPU:
    return PAGE_GPU(direction_bins=10, device=device)


def compute_pst_edge(
    bgr: np.ndarray,
    device: str = "cuda",
    params: PstEdgeParams | Mapping[str, Any] | None = None,
    pad_size: int = PAD_SIZE,
) -> np.ndarray:
    p = merge_pst_params(params)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    padded = pad_gray_reflect(gray, pad_size)

    pst = _pst_gpu(device)
    pst.h = padded.shape[0]
    pst.w = padded.shape[1]
    pst.img = torch.from_numpy(padded).to(pst.device)
    pst.img = replace_zeros_tensor(pst.img, 1)

    pst.init_kernel(p.S, p.W)
    pst.apply_kernel(
        p.sigma_LPF,
        p.thresh_min,
        p.thresh_max,
        p.morph_flag,
    )
    edge_map = pst.pst_output.cpu().numpy()
    return unpad_gray(edge_map, pad_size)


def compute_page_edge(
    bgr: np.ndarray,
    device: str = "cuda",
    params: PageEdgeParams | Mapping[str, Any] | None = None,
    pad_size: int = PAD_SIZE,
) -> np.ndarray:
    p = merge_page_params(params)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    padded = pad_gray_reflect(gray, pad_size)

    page = _page_gpu(device)
    page.h = padded.shape[0]
    page.w = padded.shape[1]
    page.img = torch.from_numpy(padded).to(page.device)
    page.img = replace_zeros_tensor(page.img, 1)

    page.init_kernel(
        p.mu_1,
        p.mu_2,
        p.sigma_1,
        p.sigma_2,
        p.S1,
        p.S2,
    )
    page.apply_kernel(
        p.sigma_LPF,
        p.thresh_min,
        p.thresh_max,
        p.morph_flag,
    )
    page.create_page_edge()
    edge_map = page.page_edge.cpu().numpy()
    return unpad_gray(edge_map, pad_size)
