"""Helpers shared by repo-root FastBlend CLI scripts."""

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from ezsynth.utils.io_utils import get_sorted_image_paths, read_image, write_image
from FastBlend.src.api import create_config
from FastBlend.src.config import FastBlendConfig
from FastBlend.src.engine import FastBlendEngine, FastBlendInput_Keyframes

_STANDALONE_DEFAULTS = {
    "enabled": True,
    "accuracy": 2,
    "window_size": 5,
    "batch_size": 16,
    "minimum_patch_size": 5,
    "num_iter": 5,
    "guide_weight": 10.0,
    "backend": "auto",
}

_INTERPOLATION_DEFAULTS = {
    "enabled": True,
    "accuracy": 2,
    "window_size": 5,
    "batch_size": 8,
    "minimum_patch_size": 15,
    "num_iter": 5,
    "guide_weight": 10.0,
    "backend": "auto",
}


def create_fastblend_config(fastblend_params: Optional[dict] = None) -> FastBlendConfig:
    """FastBlend config for full-sequence post-processing."""
    params = dict(_STANDALONE_DEFAULTS)
    if fastblend_params:
        params.update(fastblend_params)

    return create_config(
        accuracy=params["accuracy"],
        window_size=params["window_size"],
        batch_size=params["batch_size"],
        minimum_patch_size=params["minimum_patch_size"],
        num_iter=params["num_iter"],
        guide_weight=params["guide_weight"],
        backend=params["backend"],
    )


def create_interpolation_config(
    fastblend_params: Optional[dict] = None,
) -> FastBlendConfig:
    """FastBlend config tuned for keyframe interpolation."""
    params = dict(_INTERPOLATION_DEFAULTS)
    if fastblend_params:
        params.update(fastblend_params)

    return create_config(
        accuracy=params["accuracy"],
        window_size=params["window_size"],
        batch_size=params["batch_size"],
        minimum_patch_size=params["minimum_patch_size"],
        num_iter=params["num_iter"],
        guide_weight=params["guide_weight"],
        backend=params["backend"],
    )


def load_frames_sorted_with_names(
    directory: str,
    *,
    tqdm_desc: Optional[str] = None,
) -> tuple[List[np.ndarray], List[str]]:
    """Load frames by natural image sort and return aligned basenames."""
    path = Path(directory)
    paths = get_sorted_image_paths(path)
    desc = tqdm_desc or f"Loading frames from {path.name}"
    frames = [read_image(p) for p in tqdm(paths, desc=desc)]
    names = [p.name for p in paths]
    print(f"Loaded {len(frames)} frames from {directory}")
    return frames, names


def load_frames_sorted(directory: str) -> List[np.ndarray]:
    """Load frames from ``directory`` using natural image sort."""
    frames, _ = load_frames_sorted_with_names(directory)
    return frames


def save_frames(
    frames: List[np.ndarray],
    output_dir: str,
    prefix: str = "fastblend_",
) -> None:
    """Write ``{prefix}{i:05d}.png`` under ``output_dir``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(frames)} frames to {output_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        output_filename = f"{prefix}{i:05d}.png"
        write_image(output_path / output_filename, frame)


def interpolate_frames(
    guide_frames: List[np.ndarray],
    style_frames: List[np.ndarray],
    keyframe_indices: List[int],
    config: FastBlendConfig,
    progress_callback: Optional[Callable[..., None]] = None,
) -> List[np.ndarray]:
    """Interpolate between keyframes using FastBlend patch matching."""
    if len(keyframe_indices) == 0:
        raise ValueError("No keyframes found")

    if len(keyframe_indices) == 1:
        single_keyframe = style_frames[0]
        return [single_keyframe] * len(guide_frames)

    if len(keyframe_indices) == len(guide_frames):
        result_frames = [None] * len(guide_frames)
        for i, kf_idx in enumerate(keyframe_indices):
            if i < len(style_frames):
                result_frames[kf_idx] = style_frames[i]
        return result_frames  # type: ignore[return-value]

    engine = FastBlendEngine()
    return engine.interpolate_keyframes(
        FastBlendInput_Keyframes(guide_frames, style_frames, keyframe_indices),
        config,
        progress_callback,
    )
