"""Precompute state and cache-aware guide computation helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .utils.pipeline_utils import has_exact_cache_files


@dataclass
class PrecomputeState:
    """Computed data shared by synthesis passes."""

    edge_maps: List[np.ndarray] = field(default_factory=list)
    fwd_flows: List[np.ndarray] = field(default_factory=list)
    bwd_flows: List[np.ndarray] = field(default_factory=list)
    fwd_occlusion_masks: List[np.ndarray] = field(default_factory=list)
    bwd_occlusion_masks: List[np.ndarray] = field(default_factory=list)
    sparse_guides: List[np.ndarray] = field(default_factory=list)


def compute_guide(
    content_frames: List[np.ndarray],
    cache_dir: str,
    prefix: str,
    extension: str,
    force_recompute: bool,
    num_expected: int,
    title: str,
    load_cache_fn: Callable,
    compute_fn: Callable,
    save_fn: Callable,
    compute_kwargs: Optional[dict] = None,
):
    cache_path = Path(cache_dir) / prefix
    if not force_recompute and has_exact_cache_files(
        cache_path, extension, num_expected
    ):
        result = load_cache_fn(cache_path)
    else:
        result = compute_fn(content_frames, **(compute_kwargs or {}))
        save_fn(result, cache_path)

    print(f"{title} pre-computation finished.")
    return result
