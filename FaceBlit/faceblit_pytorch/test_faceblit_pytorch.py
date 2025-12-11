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


def log(msg: str) -> None:
    print(f"[test] {msg}")


def main() -> None:
    examples_dir = ROOT / "examples"
    models_dir = ROOT / "models"
    output_dir = ROOT / "faceblit_pytorch" / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # FB_DFS_MODE options:
    #   auto   -> prefer numba DFS when available; else python DFS
    #   python -> original deque DFS (deterministic)
    #   numba  -> JIT DFS if numba present (falls back to python on failure)
    dfs_mode = "numba"
    log(f"Using device: {device}, dfs_mode: {dfs_mode}")

    input_path = examples_dir / "target2.png"
    style_path = examples_dir / "style_watercolorgirl.png"
    predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"

    for p in (input_path, style_path, predictor_path):
        if not p.exists():
            log(f"Missing required asset: {p}")
            return

    # Try importing dlib for landmark detection
    try:
        import dlib  # type: ignore
    except ImportError:
        log("dlib not installed; skipping test.")
        return

    log("Running style precompute (reuse if already cached)...")
    t0 = time.perf_counter()
    style_assets = fb.compute_style_assets(
        input_path=style_path,
        output_dir=output_dir,
        predictor_path=predictor_path,
        draw_grid=False,
        stretch_hist=True,
        reuse_precomputed=True,
        device=device,
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
    engine = fb.FaceBlit(device=device)
    engine.load_style_with_guides(
        str(style_path),
        str(landmarks_path),
        str(lut_path),
        str(style_pos_path),
        str(style_app_path),
    )

    log("Running dlib landmark detection on target...")
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(str(predictor_path))
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    faces = detector(target_img_rgb, 1)
    if len(faces) == 0:
        log("No face detected in target image.")
        return
    shape = sp(target_img_rgb, faces[0])
    target_landmarks = [
        (shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)
    ]

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
