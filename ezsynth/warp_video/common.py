"""Shared helpers for flow-based video warping scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import PrecomputationConfig
from ..flow.types import FlowEngineName, FlowModelName
from ..utils.warp_utils import Warp


def load_video_frames(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    """Read up to ``num_frames`` BGR frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return frames


def get_style_keyframes(style_dir: str | Path) -> dict[int, str]:
    """Map frame index -> path for ``styleNNN.{jpg,png,jpeg}`` files."""
    style_dir = Path(style_dir)
    styles: dict[int, str] = {}
    if not style_dir.is_dir():
        return styles
    pattern = re.compile(r"style(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)
    for path in style_dir.iterdir():
        match = pattern.search(path.name)
        if match:
            styles[int(match.group(1))] = str(path)
    return styles


def precomputation_config_for_engine(
    engine: FlowEngineName,
    flow_model: FlowModelName | None = None,
) -> PrecomputationConfig:
    """Build flow precompute settings with a sensible default checkpoint per engine."""
    if flow_model is None:
        flow_model = "sintel" if engine == "RAFT" else "neuflow_mixed"
    return PrecomputationConfig(flow_engine=engine, flow_model=flow_model)


def write_png_sequence(frames: list[np.ndarray], output_dir: str | Path) -> Path:
    """Write ``{i:05d}.png`` frames; returns the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(output_dir / f"{i:05d}.png"), frame)
    return output_dir


def sample_flow_bilinear(flow: np.ndarray, pos_xy: np.ndarray) -> np.ndarray:
    """Sample HxWx2 flow at float (x, y) positions."""
    map_x = pos_xy[..., 0].astype(np.float32)
    map_y = pos_xy[..., 1].astype(np.float32)
    fx = cv2.remap(
        flow[..., 0],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    fy = cv2.remap(
        flow[..., 1],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return np.stack([fx, fy], axis=-1)


def compose_adjacent_flows(flow_segments: list[np.ndarray]) -> np.ndarray:
    """Compose forward flows along a path (each segment: current frame -> next)."""
    if not flow_segments:
        raise ValueError("compose_adjacent_flows: empty segment list")
    h, w = flow_segments[0].shape[:2]
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing="xy",
    )
    pos = np.stack([x, y], axis=-1)
    total = np.zeros_like(pos)
    for segment in flow_segments:
        sampled = sample_flow_bilinear(segment, pos)
        total += sampled
        pos = pos + sampled
    return total.astype(np.float32)


def flow_between_frames(
    k: int,
    i: int,
    adj_fwd: list[np.ndarray],
    adj_to_prev: list[np.ndarray],
    h: int,
    w: int,
) -> np.ndarray:
    """Optical flow from frame k to frame i using precomputed adjacent flows."""
    if k == i:
        return np.zeros((h, w, 2), dtype=np.float32)
    if k < i:
        return compose_adjacent_flows([adj_fwd[j] for j in range(k, i)])
    return compose_adjacent_flows([adj_to_prev[j] for j in range(k - 1, i - 1, -1)])


def precompute_adjacent_flows(
    frames: list[np.ndarray],
    compute_flow: Callable[[list[np.ndarray]], list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns (adj_fwd, adj_to_prev).

    adj_fwd[j]: flow frames[j] -> frames[j+1]
    adj_to_prev[j]: flow frames[j+1] -> frames[j]
    """
    n = len(frames)
    if n < 2:
        return [], []
    adj_fwd = compute_flow(frames)
    rev_flows = compute_flow(list(reversed(frames)))
    adj_to_prev = [rev_flows[n - 2 - j] for j in range(n - 1)]
    return adj_fwd, adj_to_prev


def compute_warp_score_from_flow(
    warper: Warp, src_frame: np.ndarray, tgt_frame: np.ndarray, flow: np.ndarray
) -> float:
    """PSNR of backward-warped content vs target (higher = better alignment)."""
    warped_content = warper.run_warping(src_frame, -flow)
    mse = np.mean(
        (warped_content.astype(np.float32) - tgt_frame.astype(np.float32)) ** 2
    )
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0


def blended_splat_fill_holes(
    warper: Warp,
    w0: np.ndarray,
    w1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    tw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize weighted blend of two forward splats and fill holes.

    Returns (filled_bgr, hole_mask) where hole_mask is uint8 255 on splat holes.
    """
    raw = (
        w0.astype(np.float32) * c0[..., np.newaxis]
        + w1.astype(np.float32) * c1[..., np.newaxis]
    )
    wmap = tw.astype(np.float32).copy()
    if warper.use_taichi and getattr(warper, "_taichi_available", False):
        warper.run_pull_push(raw, wmap)
    div = np.maximum(wmap[..., np.newaxis], 1e-6)
    filled = (raw / div).clip(0, 255).astype(np.uint8)
    residual = (tw < 1e-5).astype(np.uint8) * 255
    if np.any(residual):
        filled = cv2.inpaint(filled, residual, 4, cv2.INPAINT_TELEA)
    return filled, residual
