"""Helpers shared by repo-root FastBlend CLI scripts."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ezsynth.utils.io_utils import get_sorted_image_paths, read_image, write_image
from FastBlend.src.api import create_config
from FastBlend.src.config import FastBlendConfig
from FastBlend.src.engine import FastBlendEngine, FastBlendInput_Keyframes

logger = logging.getLogger(__name__)

# ``create_config`` keyword defaults of the CLI scripts and the ComfyUI node.
_STANDALONE_DEFAULTS = {
    "accuracy": 2,
    "window_size": 5,
    "batch_size": 16,
    "minimum_patch_size": 5,
    "num_iter": 5,
    "guide_weight": 10.0,
    "backend": "auto",
}

_INTERPOLATION_DEFAULTS = {
    **_STANDALONE_DEFAULTS,
    "batch_size": 8,
    "minimum_patch_size": 15,
}


def create_fastblend_config(fastblend_params: dict | None = None) -> FastBlendConfig:
    """FastBlend config for full-sequence post-processing."""
    return create_config(**{**_STANDALONE_DEFAULTS, **(fastblend_params or {})})


def create_interpolation_config(
    fastblend_params: dict | None = None,
) -> FastBlendConfig:
    """FastBlend config tuned for keyframe interpolation."""
    return create_config(**{**_INTERPOLATION_DEFAULTS, **(fastblend_params or {})})


def load_frames_sorted_with_names(
    directory: str,
) -> tuple[list[np.ndarray], list[str]]:
    """Load frames by natural image sort and return aligned basenames."""
    path = Path(directory)
    paths = get_sorted_image_paths(path)
    frames = [
        read_image(p) for p in tqdm(paths, desc=f"Loading frames from {path.name}")
    ]
    names = [p.name for p in paths]
    logger.info(f"Loaded {len(frames)} frames from {directory}")
    return frames, names


def load_frames_sorted(directory: str) -> list[np.ndarray]:
    """Load frames from ``directory`` using natural image sort."""
    frames, _ = load_frames_sorted_with_names(directory)
    return frames


def save_frames(
    frames: list[np.ndarray],
    output_dir: str,
    prefix: str = "fastblend_",
) -> None:
    """Write ``{prefix}{i:05d}.png`` under ``output_dir``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(frames)} frames to {output_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        output_filename = f"{prefix}{i:05d}.png"
        write_image(output_path / output_filename, frame)


def interpolate_frames(
    guide_frames: list[np.ndarray],
    style_frames: list[np.ndarray],
    keyframe_indices: list[int],
    config: FastBlendConfig,
    progress_callback: Callable[..., None] | None = None,
) -> list[np.ndarray]:
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
