"""Dispatch edge extraction by method and normalize output for the pipeline."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .classic import compute_classic_edge
from .ops import postprocess_pst_page, to_bgr_three_channel
from .phycv import compute_page_edge, compute_pst_edge

EdgeMethod = Literal["Classic", "PAGE", "PST"]


def compute_edge_frame(
    frame: np.ndarray,
    method: EdgeMethod,
    device: str = "cuda",
) -> np.ndarray:
    """Single-frame edge map as uint8 BGR."""
    if method == "Classic":
        edge_map = compute_classic_edge(frame)
    elif method == "PST":
        edge_map = compute_pst_edge(frame, device)
        edge_map = postprocess_pst_page(edge_map)
    elif method == "PAGE":
        edge_map = compute_page_edge(frame, device)
        edge_map = postprocess_pst_page(edge_map)
    else:
        raise ValueError(f"Unknown edge method: {method!r}")

    return to_bgr_three_channel(edge_map)


def compute_edge(
    frames: Sequence[np.ndarray],
    edge_method: EdgeMethod,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Edge maps for a frame sequence."""
    return [compute_edge_frame(frame, edge_method, device) for frame in frames]
