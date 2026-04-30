"""Draw 68-point face landmarks and encode frame folders as preview MP4s."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

# Open polyline index lists for iBUG 68 / dlib ordering
_DLIB_68_OPEN_CHAINS: tuple[list[int], ...] = (
    list(range(0, 17)),  # jaw
    list(range(17, 22)),  # right brow
    list(range(22, 27)),  # left brow
    list(range(27, 31)),  # nose bridge
    list(range(31, 36)),  # nose lower
)
# Closed contours (eyes, lips)
_DLIB_68_CLOSED: tuple[list[int], ...] = (
    list(range(36, 42)),
    list(range(42, 48)),
    list(range(48, 60)),
    list(range(60, 68)),
)


def natural_sort_paths(paths: list[Path]) -> list[Path]:
    """Sort paths by embedded integers in the filename (e.g. 2.png before 10.png)."""

    def key(p: Path) -> list:
        return [
            int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.stem)
        ]

    return sorted(paths, key=key)


def list_frame_paths(frames_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return image files under ``frames_dir`` (non-recursive), naturally sorted."""
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(frames_dir.glob(f"*{ext}"))
        paths.extend(frames_dir.glob(f"*{ext.upper()}"))
    # de-dupe same file matched by two cases
    uniq = {p.resolve(): p for p in paths}
    return natural_sort_paths(list(uniq.values()))


def draw_landmarks_68_bgr(
    image_bgr: np.ndarray,
    landmarks: Sequence[tuple[int, int]],
    *,
    line_bgr: tuple[int, int, int] = (255, 0, 0),  # blue outline (BGR)
    point_bgr: tuple[int, int, int] = (0, 0, 255),  # red points (BGR)
    line_thickness: int = 1,
    point_radius: int = 2,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with 68-point mesh and points drawn."""
    if len(landmarks) != 68:
        raise ValueError(f"Expected 68 landmarks, got {len(landmarks)}")
    out = image_bgr.copy()
    pts = np.asarray(landmarks, dtype=np.int32).reshape(-1, 1, 2)

    for chain in _DLIB_68_OPEN_CHAINS:
        segment = pts[chain]
        cv2.polylines(
            out,
            [segment],
            isClosed=False,
            color=line_bgr,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    for chain in _DLIB_68_CLOSED:
        segment = pts[chain]
        cv2.polylines(
            out,
            [segment],
            isClosed=True,
            color=line_bgr,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    for x, y in landmarks:
        cv2.circle(
            out, (int(x), int(y)), point_radius, point_bgr, -1, lineType=cv2.LINE_AA
        )
    return out


def _put_text_black_white_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    inner_thickness: int = 1,
    outline_ring: int = 2,
) -> None:
    """Draw ``text`` in black with a white outer stroke (BGR, ``LINE_AA``)."""
    outline_bgr = (255, 255, 255)
    fill_bgr = (0, 0, 0)
    t_outline = inner_thickness + 2 * outline_ring
    cv2.putText(
        img, text, org, font_face, font_scale, outline_bgr, t_outline, cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, font_face, font_scale, fill_bgr, inner_thickness, cv2.LINE_AA
    )


def _ffmpeg_binary(ffmpeg_path: Path | None) -> str:
    if ffmpeg_path is not None:
        return str(ffmpeg_path)
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError(
            "ffmpeg not found on PATH; install it (e.g. brew install ffmpeg) "
            "for H.264 output that plays in browsers, or pass encode_for_browser=False."
        )
    return found


def remux_to_browser_h264_mp4(
    input_mp4: Path,
    output_mp4: Path,
    *,
    ffmpeg_path: Path | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> None:
    """Re-encode ``input_mp4`` to H.264 + yuv420p + faststart (HTML5-friendly)."""
    ffmpeg = _ffmpeg_binary(ffmpeg_path)
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
        str(crf),
        "-preset",
        preset,
        str(output_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg H.264 encode failed (exit {proc.returncode}): {err or 'no stderr'}"
        )


def _row_has_valid_landmarks(row: np.ndarray) -> bool:
    """True if ``row`` is a finite (68,2) pose (not pre-face padding)."""
    if not np.isfinite(row).all():
        return False
    return float(np.max(np.abs(row))) > 1e-6


def _valid_pose_rows_mask(pts: np.ndarray) -> np.ndarray:
    """(T,) bool: finite (68,2) rows with non-zero norm (excludes pre-face ``nan`` rows)."""
    fin = np.isfinite(pts).all(axis=(1, 2))
    amp = np.nanmax(np.abs(pts), axis=(1, 2))
    return fin & (amp > 1e-6)


def _temporal_gaussian_smooth_points(
    pts: np.ndarray, sigma_frames: float
) -> np.ndarray:
    """Smooth ``(T, 68, 2)`` landmark tracks over time.

    Only smooths contiguous **valid** rows (finite coordinates). Invalid rows
    (e.g. ``nan`` before the first detection) stay invalid. Uses edge padding
    inside each smoothed segment so the first/last frames are not pulled toward
    zero (avoids corner artifacts).
    """
    if sigma_frames <= 0:
        return pts.copy()
    t, n_lm, d = pts.shape
    valid = _valid_pose_rows_mask(pts)
    if not valid.any():
        return pts.copy()

    a = int(np.argmax(valid))
    rev = valid[::-1]
    b = t - 1 - int(np.argmax(rev))

    r = max(1, int(np.ceil(3.0 * sigma_frames)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma_frames * sigma_frames))
    k /= k.sum()

    out = np.full_like(pts, np.nan, dtype=np.float64)
    seg = pts[a : b + 1].reshape(b - a + 1, -1)
    seg_sm = np.empty_like(seg)
    for j in range(seg.shape[1]):
        sig = seg[:, j]
        ext = np.pad(sig, (r, r), mode="edge")
        seg_sm[:, j] = np.convolve(ext, k, mode="valid")
    out[a : b + 1] = seg_sm.reshape(b - a + 1, n_lm, d)
    return out


def _per_row_centroids(stack: np.ndarray) -> np.ndarray:
    """(T, 2) centroid per frame; ``nan`` when that row has no valid pose."""
    t = stack.shape[0]
    out = np.full((t, 2), np.nan, dtype=np.float64)
    for i in range(t):
        row = stack[i]
        if _row_has_valid_landmarks(row):
            out[i] = row.mean(axis=0)
    return out


def _similarity_fit_prev_to_curr(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Return predicted ``curr`` from ``prev`` under 2D similarity (Umeyama, row points)."""
    mu_p = prev.mean(axis=0)
    mu_c = curr.mean(axis=0)
    x = prev - mu_p
    y = curr - mu_c
    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[1, :] *= -1.0
        r = u @ vt
    var_x = float(np.sum(x * x))
    if var_x < 1e-12:
        return curr.copy()
    scal = float(np.trace(r @ x.T @ y) / var_x)
    return scal * (x @ r.T) + mu_c


def _pairwise_rigid_coherence_series(raw: np.ndarray) -> np.ndarray:
    """Per-frame score in ``[0, 1]``: how well curr vs prev is explained by similarity."""
    t = raw.shape[0]
    out = np.zeros(t, dtype=np.float64)
    for i in range(1, t):
        if not (
            _row_has_valid_landmarks(raw[i - 1]) and _row_has_valid_landmarks(raw[i])
        ):
            continue
        pred = _similarity_fit_prev_to_curr(raw[i - 1], raw[i])
        err = float(np.mean(np.linalg.norm(raw[i] - pred, axis=1)))
        mag = float(np.mean(np.linalg.norm(raw[i] - raw[i - 1], axis=1)))
        out[i] = float(np.clip(1.0 - err / (mag + 1e-8), 0.0, 1.0))
    return out


def _smooth_landmarks_for_video(
    raw: np.ndarray,
    sigma_frames: float,
    *,
    motion_aware: bool = True,
    follow_sigma_scale: float = 0.22,
    rigid_mix_power: float = 1.35,
    follow_sigma_min: float = 0.35,
) -> np.ndarray:
    """Temporal smooth in **centroid-relative** space, then add raw centroids back.

    When ``motion_aware`` is True, blends toward a lighter smooth when consecutive
    frames look like a similarity transform (collective motion), so fast pans /
    turns are not over-damped while local jitter is still reduced.
    """
    if sigma_frames <= 0:
        return raw.copy()
    c = _per_row_centroids(raw)
    rel = raw - c[:, np.newaxis, :]
    if not motion_aware:
        rel_out = _temporal_gaussian_smooth_points(rel, sigma_frames)
        return rel_out + c[:, np.newaxis, :]

    s_hi = sigma_frames
    s_lo = min(s_hi, max(follow_sigma_min, sigma_frames * follow_sigma_scale))
    rel_hi = _temporal_gaussian_smooth_points(rel, s_hi)
    rel_lo = _temporal_gaussian_smooth_points(rel, s_lo)
    coh = _pairwise_rigid_coherence_series(raw)
    vm = _valid_pose_rows_mask(raw).astype(np.float64)
    mix = (coh**rigid_mix_power)[:, np.newaxis, np.newaxis] * vm[
        :, np.newaxis, np.newaxis
    ]
    rel_out = (1.0 - mix) * rel_hi + mix * rel_lo
    return rel_out + c[:, np.newaxis, :]


def _collect_carried_landmark_sequence(
    frame_paths: list[Path],
    w: int,
    h: int,
    detect_landmarks,
    landmark_backend: str,
    predictor: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(stack, detected)``: (T,68,2) carried coords and (T,) True if detected this frame."""
    rows: list[np.ndarray] = []
    detected: list[bool] = []
    last_lm: np.ndarray | None = None
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"Could not read image: {path}")
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        lm = detect_landmarks(img, landmark_backend, predictor, device)
        det = lm is not None and len(lm) == 68
        detected.append(det)
        if det:
            last_lm = np.asarray(lm, dtype=np.float64)
        if last_lm is not None:
            rows.append(last_lm.copy())
        else:
            rows.append(np.full((68, 2), np.nan, dtype=np.float64))
    return np.stack(rows, axis=0), np.array(detected, dtype=bool)


def _landmarks_to_int_tuples(lm: np.ndarray) -> list[tuple[int, int]]:
    return [(int(round(lm[i, 0])), int(round(lm[i, 1]))) for i in range(68)]


# Representative dlib-68 indices for trajectory plots (jaw, brow, nose, eye, mouth)
_DEFAULT_TRAJECTORY_LM_INDICES: tuple[int, ...] = (
    0,
    8,
    17,
    22,
    27,
    30,
    36,
    45,
    48,
    54,
)


def write_landmark_trajectory_figure(
    raw_stack: np.ndarray,
    smooth_stack: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    landmark_indices: Sequence[int] | None = None,
    title: str | None = None,
) -> Path:
    """Save a figure: X vs time and Y vs time with raw (dashed) and smoothed (solid).

    Time axis is frame index / ``fps`` (seconds). Invalid (pre-face) samples are
    masked. Requires matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(raw_stack.shape[0], dtype=np.float64) / float(fps)
    indices = (
        tuple(landmark_indices)
        if landmark_indices is not None
        else _DEFAULT_TRAJECTORY_LM_INDICES
    )
    for idx in indices:
        if idx < 0 or idx >= 68:
            raise ValueError(f"landmark index out of range [0,67]: {idx}")

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
    )
    n_idx = len(indices)
    cm = plt.get_cmap("tab10")
    for j, idx in enumerate(indices):
        color = cm(j / max(n_idx - 1, 1))
        for ax, dim in ((ax0, 0), (ax1, 1)):
            raw_c = np.asarray(raw_stack[:, idx, dim], dtype=np.float64)
            sm_c = np.asarray(smooth_stack[:, idx, dim], dtype=np.float64)
            raw_m = np.ma.masked_where(~np.isfinite(raw_c), raw_c)
            sm_m = np.ma.masked_where(~np.isfinite(sm_c), sm_c)
            ax.plot(
                t,
                raw_m,
                "--",
                color=color,
                alpha=0.65,
                lw=1.0,
            )
            ax.plot(
                t,
                sm_m,
                "-",
                color=color,
                alpha=0.95,
                lw=1.35,
                label=(f"#{idx}" if ax is ax0 else None),
            )
    ax0.set_ylabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.set_xlabel("Time (s)")
    h0, l0 = ax0.get_legend_handles_labels()
    ax0.legend(
        h0,
        l0,
        loc="upper right",
        fontsize=8,
        ncol=2,
        framealpha=0.9,
        title="Solid=smoothed, dashed=raw (same color per #)",
    )
    fig.suptitle(title or "Landmark coordinates vs time (image space)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _gather_landmarks_stack(
    frames_dir: Path,
    extensions: tuple[str, ...],
    predictor_path: Path,
    landmark_backend: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[Path], int, int]:
    """Load frames, run detection, return ``(raw_stack, detected_mask, paths, h, w)``."""
    from src.utils_face import detect_landmarks

    frames_dir = Path(frames_dir)
    frame_paths = list_frame_paths(frames_dir, extensions)
    if not frame_paths:
        raise FileNotFoundError(f"No images matching {extensions} in {frames_dir}")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise OSError(f"Could not read image: {frame_paths[0]}")
    h, w = first.shape[:2]
    raw_stack, detected_mask = _collect_carried_landmark_sequence(
        frame_paths,
        w,
        h,
        detect_landmarks,
        landmark_backend,
        predictor_path,
        device,
    )
    return raw_stack, detected_mask, frame_paths, h, w


def write_landmarks_preview_mp4(
    frames_dir: Path,
    output_mp4: Path,
    *,
    predictor_path: Path,
    device: str = "cpu",
    landmark_backend: str = "dlib",
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
    faceblit_taichi_triplet: bool = False,
    faceblit_taichi_solo: bool = False,
    faceblit_style_path: Path | None = None,
    faceblit_style_cache_dir: Path | None = None,
) -> Path:
    """Detect landmarks on each frame in ``frames_dir`` and save an MP4 at ``output_mp4``.

    By default the file is H.264 (yuv420p, ``+faststart``) via ffmpeg so it plays in
    browsers. OpenCV's ``mp4v`` is only used as an intermediate when transcoding.

    If ``smooth_landmarks`` is True, output is side-by-side: left = carried raw
    detections, right = temporally smoothed tracks (same dimensions per half).
    Smoothing is **centroid-relative** (global translation kept from raw) and,
    when ``smooth_motion_aware`` is True, reduces damping when motion matches a
    2D similarity between consecutive frames. Outline is blue, landmark dots red.

    If ``faceblit_taichi_solo`` is True (requires ``smooth_landmarks``), each frame
    is only the Taichi FaceBlit result (``stylize_background=False``) at the
    original resolution—no landmark overlay columns.

    If ``faceblit_taichi_triplet`` is True (requires ``smooth_landmarks``; mutually
    exclusive with ``faceblit_taichi_solo``), two landmark columns plus a third
    column: stylization using ``faceblit_style_path`` and **smoothed** landmarks.
    """
    frames_dir = Path(frames_dir)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    raw_stack, detected_mask, frame_paths, h, w = _gather_landmarks_stack(
        frames_dir,
        extensions,
        predictor_path,
        landmark_backend,
        device,
    )

    smooth_seq: np.ndarray | None = None
    if smooth_landmarks:
        smooth_seq = _smooth_landmarks_for_video(
            raw_stack,
            smooth_sigma_frames,
            motion_aware=smooth_motion_aware,
            follow_sigma_scale=smooth_follow_sigma_scale,
            rigid_mix_power=smooth_rigid_mix_power,
        )

    smooth_for_plot = smooth_seq
    if trajectory_plot_path is not None and smooth_for_plot is None:
        smooth_for_plot = _smooth_landmarks_for_video(
            raw_stack,
            smooth_sigma_frames,
            motion_aware=smooth_motion_aware,
            follow_sigma_scale=smooth_follow_sigma_scale,
            rigid_mix_power=smooth_rigid_mix_power,
        )
    if trajectory_plot_path is not None and smooth_for_plot is not None:
        write_landmark_trajectory_figure(
            raw_stack,
            smooth_for_plot,
            trajectory_plot_path,
            fps=fps,
            landmark_indices=trajectory_plot_indices,
        )

    if faceblit_taichi_triplet and faceblit_taichi_solo:
        raise ValueError(
            "faceblit_taichi_triplet and faceblit_taichi_solo are mutually exclusive"
        )
    if faceblit_taichi_triplet and not smooth_landmarks:
        raise ValueError("faceblit_taichi_triplet requires smooth_landmarks=True")
    if faceblit_taichi_solo and not smooth_landmarks:
        raise ValueError("faceblit_taichi_solo requires smooth_landmarks=True")

    n_cols = 1
    if smooth_landmarks:
        if faceblit_taichi_solo:
            n_cols = 1
        else:
            n_cols = 3 if faceblit_taichi_triplet else 2
    out_w = w * n_cols
    out_h = h

    fb_engine = None
    fb_style = None
    if faceblit_taichi_triplet or faceblit_taichi_solo:
        from src.engine import FaceBlitEngine

        style_p = Path(faceblit_style_path or _default_faceblit_style_path())
        if not style_p.is_file():
            raise FileNotFoundError(f"FaceBlit style image not found: {style_p}")
        cache = Path(
            faceblit_style_cache_dir
            or (output_mp4.parent / f"_fb_style_cache_{style_p.stem}")
        )
        print(f"[landmark viz] Precomputing Taichi style assets → {cache}")
        fb_style = _prepare_faceblit_style_taichi(
            style_p,
            cache,
            landmark_backend=landmark_backend,
            predictor_path=predictor_path,
            device=device,
        )
        fb_engine = FaceBlitEngine()

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4, " "))
    if encode_for_browser:
        fd, raw_path = tempfile.mkstemp(suffix=".mp4", prefix="faceblit_landmarks_")
        os.close(fd)
        video_target = Path(raw_path)
        cleanup_raw = True
    else:
        video_target = output_mp4
        cleanup_raw = False

    writer = cv2.VideoWriter(str(video_target), fourcc_code, fps, (out_w, out_h))
    if not writer.isOpened():
        if encode_for_browser:
            video_target.unlink(missing_ok=True)
        raise RuntimeError(
            f"VideoWriter failed to open for {video_target} "
            f"(codec {fourcc!r}, size {out_w}x{out_h})"
        )

    try:
        if faceblit_taichi_solo:
            desc = "[faceblit taichi]"
        elif faceblit_taichi_triplet:
            desc = "[viz+taichi]"
        else:
            desc = "[landmark viz]"
        for i, path in enumerate(tqdm(frame_paths, desc=desc)):
            img = cv2.imread(str(path))
            if img is None:
                raise OSError(f"Could not read image: {path}")
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

            row = raw_stack[i]
            has_pts = _row_has_valid_landmarks(row)
            if has_pts:
                left_vis = draw_landmarks_68_bgr(img, _landmarks_to_int_tuples(row))
                if not detected_mask[i]:
                    _put_text_black_white_outline(
                        left_vis, "no face (last)", (8, 24), font_scale=0.7
                    )
            else:
                left_vis = img.copy()
                _put_text_black_white_outline(
                    left_vis, "no face", (8, 24), font_scale=0.7
                )

            if smooth_landmarks and smooth_seq is not None:
                sm = smooth_seq[i]
                if (
                    faceblit_taichi_solo
                    and fb_engine is not None
                    and fb_style is not None
                ):
                    if _row_has_valid_landmarks(sm):
                        lm_tuples = _landmarks_to_int_tuples(sm)
                        vis = _stylize_target_taichi_no_bg(
                            img,
                            lm_tuples,
                            engine=fb_engine,
                            input_style=fb_style,
                            device=device,
                        )
                    else:
                        vis = img.copy()
                else:
                    _put_text_black_white_outline(
                        left_vis, "raw", (8, out_h - 12), font_scale=0.65
                    )
                    if _row_has_valid_landmarks(sm):
                        right_vis = draw_landmarks_68_bgr(
                            img, _landmarks_to_int_tuples(sm)
                        )
                    else:
                        right_vis = img.copy()
                        _put_text_black_white_outline(
                            right_vis, "no face", (8, 24), font_scale=0.7
                        )
                    _put_text_black_white_outline(
                        right_vis, "smoothed", (8, out_h - 12), font_scale=0.65
                    )
                    panels = [left_vis, right_vis]
                    if (
                        faceblit_taichi_triplet
                        and fb_engine is not None
                        and fb_style is not None
                    ):
                        if _row_has_valid_landmarks(sm):
                            lm_tuples = _landmarks_to_int_tuples(sm)
                            fb_vis = _stylize_target_taichi_no_bg(
                                img,
                                lm_tuples,
                                engine=fb_engine,
                                input_style=fb_style,
                                device=device,
                            )
                        else:
                            fb_vis = img.copy()
                            _put_text_black_white_outline(
                                fb_vis, "no face", (8, 24), font_scale=0.7
                            )
                        _put_text_black_white_outline(
                            fb_vis,
                            "FaceBlit (Taichi, no bg)",
                            (8, out_h - 12),
                            font_scale=0.55,
                        )
                        panels.append(fb_vis)
                    vis = np.hstack(panels)
            else:
                vis = left_vis

            writer.write(vis)
    finally:
        writer.release()

    if encode_for_browser:
        try:
            remux_to_browser_h264_mp4(
                video_target,
                output_mp4,
                ffmpeg_path=ffmpeg_path,
                crf=h264_crf,
                preset=h264_preset,
            )
        finally:
            if cleanup_raw:
                video_target.unlink(missing_ok=True)

    return output_mp4


def _default_predictor_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "FaceBlit" / "models" / "shape_predictor_68_face_landmarks.dat"


def _default_faceblit_style_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "FaceBlit" / "examples" / "style_watercolorgirl.png"


def _prepare_faceblit_style_taichi(
    style_path: Path,
    cache_dir: Path,
    *,
    landmark_backend: str,
    predictor_path: Path,
    device: str,
):
    """Precompute / load Taichi style LUT and guides; returns ``FaceBlitInput_Style``."""
    from src.backends.taichi_backend import ensure_ti_init
    from src.engine import FaceBlitInput_Style

    ensure_ti_init(device)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    style_path = Path(style_path)
    ap = FaceBlitInput_Style.compute_style_assets(
        cache_dir,
        style_path,
        landmark_backend,
        predictor_path,
        device,
        draw_grid=False,
        stretch_hist=True,
        backend="taichi",
    )
    return FaceBlitInput_Style.load_style_assets(
        style_path,
        ap["path_style_pos_guide"],
        ap["path_style_app_guide"],
        ap["path_style_landmarks"],
        ap["path_style_lookup_table"],
    )


def _stylize_target_taichi_no_bg(
    img_bgr: np.ndarray,
    landmarks: list[tuple[int, int]],
    *,
    engine: object,
    input_style: object,
    device: str,
) -> np.ndarray:
    """Run Taichi FaceBlit with ``stylize_background=False`` (face only, alpha blend)."""
    from src.engine import FaceBlitInput_Target

    tgt = FaceBlitInput_Target(img_bgr, landmarks)
    result, _ = engine.stylize_image_with_guide_and_landmarks(
        input_target=tgt,
        input_style=input_style,
        device=device,
        grid_size=3,
        patch_size=3,
        lambda_pos=10,
        lambda_app=2,
        use_vectorized=False,
        dfs_mode="auto",
        blend_sigma=25.0,
        stylize_background=False,
        backend="taichi",
        threshold=50,
        denoise_iters=1,
    )
    return np.asarray(result, dtype=np.uint8)


def write_landmark_trajectory_from_frames(
    frames_dir: Path,
    out_path: Path,
    *,
    predictor_path: Path,
    landmark_backend: str = "dlib",
    device: str = "cpu",
    fps: float = 25.0,
    smooth_sigma_frames: float = 2.0,
    smooth_motion_aware: bool = True,
    smooth_follow_sigma_scale: float = 0.22,
    smooth_rigid_mix_power: float = 1.35,
    landmark_indices: Sequence[int] | None = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> Path:
    """Load frames, smooth temporally, save raw vs smoothed trajectory figure only (no MP4)."""
    raw_stack, _, _, _, _ = _gather_landmarks_stack(
        frames_dir,
        extensions,
        predictor_path,
        landmark_backend,
        device,
    )
    smooth_stack = _smooth_landmarks_for_video(
        raw_stack,
        smooth_sigma_frames,
        motion_aware=smooth_motion_aware,
        follow_sigma_scale=smooth_follow_sigma_scale,
        rigid_mix_power=smooth_rigid_mix_power,
    )
    return write_landmark_trajectory_figure(
        raw_stack,
        smooth_stack,
        out_path,
        fps=fps,
        landmark_indices=landmark_indices,
    )
