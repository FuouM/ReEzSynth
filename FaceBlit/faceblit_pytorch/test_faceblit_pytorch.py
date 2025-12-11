"""
Lightweight test script for the pure Python/PyTorch FaceBlit port.

Usage:
    python test_faceblit_pytorch.py

Requires:
    - torch, numpy
    - opencv-python
    - dlib (for landmark detection; if missing, the test is skipped)
Assets:
    Uses the sample images in FaceBlit/examples and the model in FaceBlit/models.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# Make the repository root importable when running the script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faceblit_pytorch as fb  # noqa: E402

# ---------------------------------------------------------------------------
# Config toggles (edit here instead of CLI args)
# ---------------------------------------------------------------------------
LANDMARK_BACKEND = "dlib"  # "dlib" or "fan"
# dlib: very fast
# fan: very slow, wider support
FAN_MODEL = "dlib"
# dlib: fast
# sfd: best, slowest
# blazeface: front camera (doesn't seem to download)
PRECOMPUTE_DEVICE = "cuda"  # Device for style precompute operations
STYLIZE_DEVICE = "cpu"  # Device for stylization operations
DFS_MODE = "numba"  # "auto", "python", "numba"


def log(msg: str) -> None:
    print(f"[{fb.get_timestamp()}] [test] {msg}")


def detect_landmarks(
    image_bgr, backend: str, predictor_path: Path, device: str
) -> list[tuple[int, int]] | None:
    """Detect landmarks on a BGR image with the selected backend."""
    if backend == "dlib":
        try:
            import dlib  # type: ignore
        except ImportError:
            log("dlib not installed; cannot run dlib backend.")
            return None

        if not predictor_path.exists():
            log(f"Missing required predictor for dlib: {predictor_path}")
            return None

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(str(predictor_path))
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = detector(img_rgb, 1)
        if len(faces) == 0:
            log("No face detected in target image.")
            return None
        shape = sp(img_rgb, faces[0])
        return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    # FAN backend
    fa_device = "cpu"
    d = None
    try:
        import torch

        d = torch.device(device)
    except Exception:
        pass
    if d is not None:
        if d.type == "cuda":
            fa_device = "cuda"
        elif d.type == "mps":
            fa_device = "mps"

    # Use cached FAN model for better performance
    fa = fb._get_fan_model(fa_device, FAN_MODEL)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t_fan = time.perf_counter()
    preds = fa.get_landmarks(img_rgb)
    log(
        f"FAN forward pass finished in {time.perf_counter() - t_fan:.3f}s on {fa_device}"
    )
    if preds is None or len(preds) == 0:
        log("No face detected in target image (FAN).")
        return None
    shape = preds[0]
    return [(int(p[0]), int(p[1])) for p in shape]


def main() -> None:
    examples_dir = ROOT / "examples"
    models_dir = ROOT / "models"
    output_dir = ROOT / "faceblit_pytorch" / "test_fan_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    precompute_device = PRECOMPUTE_DEVICE
    stylize_device = STYLIZE_DEVICE
    dfs_mode = DFS_MODE
    landmark_backend = LANDMARK_BACKEND
    # FB_DFS_MODE options:
    #   auto   -> prefer numba DFS when available; else python DFS
    #   python -> original deque DFS (deterministic)
    #   numba  -> JIT DFS if numba present (falls back to python on failure)
    log(
        f"Using precompute_device: {precompute_device}, stylize_device: {stylize_device}, dfs_mode: {dfs_mode}, landmark_backend: {landmark_backend}"
    )

    input_path = examples_dir / "target2.png"
    style_path = examples_dir / "style_watercolorgirl.png"
    predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"

    required_assets = [input_path, style_path]
    if landmark_backend == "dlib":
        required_assets.append(predictor_path)
    for p in required_assets:
        if not p.exists():
            log(f"Missing required asset: {p}")
            return

    log("Running style precompute (reuse if already cached)...")
    t0 = time.perf_counter()
    style_assets = fb.compute_style_assets(
        input_path=style_path,
        output_dir=output_dir,
        predictor_path=predictor_path if landmark_backend == "dlib" else None,
        landmark_model=landmark_backend,
        draw_grid=False,
        stretch_hist=True,
        reuse_precomputed=True,
        device=precompute_device,
    )
    precomp_s = time.perf_counter() - t0
    log(f"Style precompute finished in {precomp_s:.3f}s")

    style_pos_path = Path(style_assets["style_pos_guide_path"])
    style_app_path = Path(style_assets["style_app_guide_path"])
    landmarks_path = Path(style_assets["landmarks_path"])
    lut_path = Path(style_assets["lut_path"])

    target_img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if target_img is None or target_img.size == 0:
        log("Failed to read target image.")
        return

    log("Preparing target appearance guide...")
    target_app_path = output_dir / "target_app.png"
    if target_app_path.exists():
        log("Loading existing target appearance guide...")
        target_app_matched = cv2.imread(str(target_app_path), cv2.IMREAD_GRAYSCALE)
        if target_app_matched is None:
            log("Failed to read existing target appearance guide.")
            return
    else:
        target_app = fb.get_app_guide(target_img, stretch_hist=False)
        ref_app = cv2.imread(str(style_app_path), cv2.IMREAD_GRAYSCALE)
        if ref_app is None:
            log("Failed to read reference style appearance guide.")
            return
        target_app_matched = fb.gray_hist_matching(target_app, ref_app)
        cv2.imwrite(str(target_app_path), target_app_matched)

    log("Loading FaceBlit engine (PyTorch port)...")
    engine = fb.FaceBlit(
        precompute_device=precompute_device, stylize_device=stylize_device
    )
    engine.load_style_with_guides(
        str(style_path),
        str(landmarks_path),
        str(lut_path),
        str(style_pos_path),
        str(style_app_path),
    )

    log(f"Running {landmark_backend} landmark detection on target...")
    target_landmarks = detect_landmarks(
        target_img, landmark_backend, predictor_path, PRECOMPUTE_DEVICE
    )
    if target_landmarks is None:
        return

    log("Stylizing...")
    t1 = time.perf_counter()
    result, target_pos_guide = engine.stylize_image_with_guide_and_landmarks(
        target_img,
        target_app_matched,
        target_landmarks,
        stylize_bg=False,
        patch_size=3,  # Use LUT voting
        dfs_mode=dfs_mode,
    )
    stylize_s = time.perf_counter() - t1
    log(f"Stylization finished in {stylize_s:.3f}s")

    output_path = output_dir / "target2_stylized_pytorch.png"
    cv2.imwrite(str(output_path), result)
    log(f"Success! Output saved to: {output_path}")

    # Save target position guide
    target_pos_path = output_dir / "target2_target_pos.png"
    cv2.imwrite(str(target_pos_path), target_pos_guide)
    log(f"Target position guide saved to: {target_pos_path}")


if __name__ == "__main__":
    main()
