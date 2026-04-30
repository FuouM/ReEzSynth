# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Literal, Sequence

# Add project roots to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FaceBlit"))

from src.utils_landmark_viz import (
    write_landmark_trajectory_from_frames,
    write_landmarks_preview_mp4,
)

PREDICTOR_PATH = ROOT / "FaceBlit" / "models" / "shape_predictor_68_face_landmarks.dat"
EXAMPLES_DIR = ROOT / "FaceBlit" / "examples"
DEFAULT_STYLE_PATH = EXAMPLES_DIR / "style_watercolorgirl.png"


def run_landmark_preview_video(
    frames_dir: Path,
    output_mp4: Path | None = None,
    *,
    landmark_backend: Literal["dlib", "fan"] = "dlib",
    device: str = "cpu",
    predictor_path: Path | None = None,
    fps: float = 25.0,
    fourcc: str = "mp4v",
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    encode_for_browser: bool = True,
    ffmpeg_path: Path | None = None,
    h264_crf: int = 20,
    h264_preset: str = "medium",
    smooth_landmarks: bool = False,
    smooth_sigma_frames: float = 2.0,
    smooth_motion_aware: bool = True,
    smooth_follow_sigma_scale: float = 0.22,
    smooth_rigid_mix_power: float = 1.35,
    trajectory_plot_path: Path | None = None,
    trajectory_plot_indices: Sequence[int] | None = None,
    trajectory_only_path: Path | None = None,
    faceblit_taichi_triplet: bool = False,
    faceblit_taichi_solo: bool = False,
    faceblit_style_path: Path | None = None,
    faceblit_style_cache_dir: Path | None = None,
    output_parent: Path | None = None,
) -> Path:
    """Landmark overlay / optional temporal smooth / optional Taichi FaceBlit column → MP4.

    If ``trajectory_only_path`` is set, writes only the matplotlib trajectory figure
    (same smooth settings) and returns that path; no video is produced.

    When ``output_mp4`` is None, defaults next to ``frames_dir`` parent:
    ``<stem>_stylized_taichi.mp4`` if ``faceblit_taichi_solo``,
    ``<stem>_landmarks_smooth_faceblit.mp4`` if ``faceblit_taichi_triplet``,
    else ``<stem>_landmarks.mp4``. If ``output_parent`` is set, files go there
    instead of ``frames_dir.parent``.
    """
    pred = predictor_path or PREDICTOR_PATH
    if landmark_backend == "dlib" and not pred.exists():
        raise FileNotFoundError(f"missing dlib predictor at {pred}")

    frames_dir = Path(frames_dir)
    do_smooth = smooth_landmarks or faceblit_taichi_triplet or faceblit_taichi_solo

    if trajectory_only_path is not None:
        path = write_landmark_trajectory_from_frames(
            frames_dir,
            trajectory_only_path,
            predictor_path=pred,
            landmark_backend=landmark_backend,
            device=device,
            fps=fps,
            smooth_sigma_frames=smooth_sigma_frames,
            smooth_motion_aware=smooth_motion_aware,
            smooth_follow_sigma_scale=smooth_follow_sigma_scale,
            smooth_rigid_mix_power=smooth_rigid_mix_power,
            landmark_indices=trajectory_plot_indices,
            extensions=extensions,
        )
        print(f"[Demo] landmark trajectory: wrote {path}")
        return path

    out: Path
    if output_mp4 is not None:
        out = Path(output_mp4)
    else:
        stem = frames_dir.name
        base = (
            output_parent if output_parent is not None else frames_dir.resolve().parent
        )
        if faceblit_taichi_solo:
            out = base / f"{stem}_stylized_taichi.mp4"
        elif faceblit_taichi_triplet:
            out = base / f"{stem}_landmarks_smooth_faceblit.mp4"
        else:
            out = base / f"{stem}_landmarks.mp4"

    path = write_landmarks_preview_mp4(
        frames_dir,
        out,
        predictor_path=pred,
        device=device,
        landmark_backend=landmark_backend,
        fps=fps,
        fourcc=fourcc,
        extensions=extensions,
        encode_for_browser=encode_for_browser,
        ffmpeg_path=ffmpeg_path,
        h264_crf=h264_crf,
        h264_preset=h264_preset,
        smooth_landmarks=do_smooth,
        smooth_sigma_frames=smooth_sigma_frames,
        smooth_motion_aware=smooth_motion_aware,
        smooth_follow_sigma_scale=smooth_follow_sigma_scale,
        smooth_rigid_mix_power=smooth_rigid_mix_power,
        trajectory_plot_path=trajectory_plot_path,
        trajectory_plot_indices=trajectory_plot_indices,
        faceblit_taichi_triplet=faceblit_taichi_triplet,
        faceblit_taichi_solo=faceblit_taichi_solo,
        faceblit_style_path=faceblit_style_path,
        faceblit_style_cache_dir=faceblit_style_cache_dir,
    )
    print(f"[Demo] landmark preview: wrote {path}")
    if trajectory_plot_path is not None:
        print(f"[Demo] trajectory plot: wrote {trajectory_plot_path}")
    return path


if __name__ == "__main__":
    video_output_dir = ROOT / "FaceBlit" / "test_output" / "video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frames = EXAMPLES_DIR / "zuzka2"

    run_landmark_preview_video(
        frames_dir=frames,
        output_mp4=video_output_dir / "zuzka2_stylized_taichi.mp4",
        landmark_backend="dlib",
        device="cuda",
        predictor_path=PREDICTOR_PATH,
        smooth_landmarks=True,
        smooth_sigma_frames=2.0,
        faceblit_taichi_triplet=True,
        faceblit_style_path=DEFAULT_STYLE_PATH,
        encode_for_browser=True,
    )
