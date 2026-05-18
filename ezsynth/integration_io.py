"""I/O helpers for external integrations.

These bridge in-memory RGB frames to the path-based pipeline without depending
on a specific UI framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .utils.io_utils import write_image


@dataclass
class FrameSequencePaths:
    directory: Path
    frame_paths: list[Path]


def rgb_float_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA float or uint frames to internal BGR uint8."""
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError(f"Expected RGB/RGBA HWC frame, got shape {frame.shape}.")

    if np.issubdtype(frame.dtype, np.floating):
        arr = np.clip(frame, 0.0, 1.0) * 255.0
        arr = np.rint(arr).astype(np.uint8)
    elif frame.dtype == np.uint8:
        arr = frame
    else:
        raise ValueError(f"Expected float or uint8 frame, got {frame.dtype}.")

    rgb = arr[..., :3]
    return rgb[..., ::-1].copy()


def bgr_uint8_to_rgb_float(frame: np.ndarray) -> np.ndarray:
    """Convert internal BGR uint8 frames to RGB float32 in ``[0, 1]``."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR HWC frame, got shape {frame.shape}.")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame.dtype}.")
    return (frame[..., ::-1].astype(np.float32) / 255.0).copy()


def write_rgb_frame_sequence(
    frames: Sequence[np.ndarray],
    directory: str | Path,
    *,
    prefix: str = "frame",
) -> FrameSequencePaths:
    """Write RGB float/uint frames as numbered PNGs and return their paths."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{i:05d}.png"
        write_image(path, rgb_float_to_bgr_uint8(frame))
        paths.append(path)
    return FrameSequencePaths(directory=out_dir, frame_paths=paths)


def write_rgb_style_images(
    styles: Sequence[np.ndarray],
    directory: str | Path,
    *,
    prefix: str = "style",
) -> list[Path]:
    """Write RGB float/uint style images and return file paths."""
    return write_rgb_frame_sequence(styles, directory, prefix=prefix).frame_paths
