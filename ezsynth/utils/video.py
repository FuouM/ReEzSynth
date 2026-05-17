"""Video frame extraction and browser-friendly H.264 export helpers."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
from tqdm import tqdm


def extract_video_frames_to_png_directory(
    video_path: Union[str, Path], output_dir: Union[str, Path]
) -> int:
    """
    Decode ``video_path`` and write ``{i:05d}.png`` into ``output_dir``.

    Returns the number of frames written.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting frames from '{video_path.name}' to '{output_dir}'...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    with tqdm(total=max(frame_count, 0), desc="Extracting Frames") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break

            output_filename = output_dir / f"{frame_num:05d}.png"
            cv2.imwrite(str(output_filename), frame)

            frame_num += 1
            pbar.update(1)

    cap.release()
    print(f"\nSuccessfully extracted {frame_num} frames.")
    return frame_num


def require_ffmpeg_binary() -> str:
    """Return the ffmpeg executable path or raise ``FileNotFoundError``."""
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError(
            "ffmpeg not found on PATH; install it (e.g. brew install ffmpeg) "
            "to export H.264 MP4 (yuv420p)."
        )
    return found


def remux_to_browser_h264_mp4(input_mp4: Path, output_mp4: Path) -> None:
    """Re-encode ``input_mp4`` to H.264 + yuv420p + faststart (HTML5-friendly)."""
    ffmpeg = require_ffmpeg_binary()
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_mp4),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "20",
        "-preset",
        "medium",
        str(output_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg H.264 encode failed (exit {proc.returncode}): {err or 'no stderr'}"
        )


def export_frames_to_browser_h264_mp4(
    frames: Sequence[np.ndarray], path: Union[str, Path], fps: float
) -> None:
    """Write ``mp4v`` to a temp file, then H.264 + yuv420p + faststart via ffmpeg."""
    if not frames:
        print("[export-mp4] No frames to encode; skipping.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc_code = cv2.VideoWriter_fourcc(*"mp4v")

    fd, raw_path = tempfile.mkstemp(suffix=".mp4", prefix="ezsynth_export_")
    os.close(fd)
    video_target = Path(raw_path)

    writer = cv2.VideoWriter(str(video_target), fourcc_code, float(fps), (w, h))
    if not writer.isOpened():
        video_target.unlink(missing_ok=True)
        raise RuntimeError(
            f"VideoWriter failed to open for {video_target} "
            f"(codec 'mp4v', size {w}x{h})"
        )
    try:
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                raise ValueError(
                    f"Frame size mismatch: expected {(h, w)}, got {frame.shape[:2]}"
                )
            bgr = frame
            if frame.dtype != np.uint8:
                bgr = np.clip(frame, 0, 255).astype(np.uint8)
            if bgr.ndim == 2:
                bgr = np.stack([bgr, bgr, bgr], axis=-1)
            if bgr.shape[2] == 4:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    try:
        remux_to_browser_h264_mp4(video_target, path)
    finally:
        video_target.unlink(missing_ok=True)
    print(
        f"Exported MP4: {path} "
        f"({len(frames)} frames @ {fps} fps, H.264 yuv420p +faststart)"
    )
