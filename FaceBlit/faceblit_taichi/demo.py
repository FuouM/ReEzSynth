import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project roots to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FaceBlit"))

from faceblit_pytorch.src import ops

from faceblit_taichi import FaceBlitTaichi


def detect_landmarks(image_bgr, predictor_path):
    import dlib

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(str(predictor_path))
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)
    if len(faces) == 0:
        return None
    shape = sp(img_rgb, faces[0])
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def main():
    examples_dir = ROOT / "FaceBlit" / "examples"
    models_dir = ROOT / "FaceBlit" / "models"
    output_dir = ROOT / "FaceBlit" / "faceblit_taichi" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor_path = models_dir / "shape_predictor_68_face_landmarks.dat"
    style_path = examples_dir / "style_watercolorgirl.png"
    target_path = examples_dir / "target2.png"

    if not predictor_path.exists():
        print(f"Error: Missing predictor at {predictor_path}")
        return

    # 1. Load Style and Precompute Assets (using existing PyTorch ops for guide gen)
    print("[Demo] Loading style and computing assets...")
    style_img = cv2.imread(str(style_path))

    # Save assets to output_dir
    style_landmarks_out = output_dir / "style_watercolorgirl_landmarks.txt"
    style_lut_out = output_dir / "style_watercolorgirl_lut.bytes"

    # Landmark Detection & Saving
    print("[Demo] Detecting style landmarks...")
    style_landmarks = detect_landmarks(style_img, predictor_path)
    if style_landmarks is not None:
        print(f"[Demo] Saving style landmarks to {style_landmarks_out}")
        with open(style_landmarks_out, "w") as f:
            f.write(f"{len(style_landmarks)}\n")
            for x, y in style_landmarks:
                f.write(f"{x} {y}\n")
    else:
        print("Error: Could not detect landmarks on style image")
        return

    style_pos = ops.gradient_guide(style_img.shape[1], style_img.shape[0])
    style_app = ops.get_app_guide(style_img, stretch_hist=True)

    # 2. Setup Taichi Engine
    fb_ti = FaceBlitTaichi()
    start_lut = time.time()
    fb_ti.load_style(style_img, style_pos, style_app)
    print(f"[Demo] Taichi LUT computation: {time.time() - start_lut:.4f}s")

    # Save LUT
    print(f"[Demo] Saving LUT to {style_lut_out}")
    ops.save_look_up_cube(
        fb_ti.look_up_cube.cpu().numpy().astype(np.uint16), style_lut_out
    )

    # 3. Process Target
    print("[Demo] Processing target...")
    target_img = cv2.imread(str(target_path))
    target_landmarks = detect_landmarks(target_img, predictor_path)
    if target_landmarks is None:
        print("Error: No face detected in target")
        return

    # MLS Warp for target pos guide
    pos_tensor = ops._to_tensor(style_pos)
    warped = ops.warp_mls_similarity(
        pos_tensor,
        torch.tensor(style_landmarks, dtype=torch.float32),
        torch.tensor(target_landmarks, dtype=torch.float32),
        grid_size=10,
    )
    target_pos = ops._to_numpy_image(warped)
    target_pos = cv2.resize(target_pos, (target_img.shape[1], target_img.shape[0]))

    # Target App Guide with Hist Matching
    target_app_raw = ops.get_app_guide(target_img, stretch_hist=False)
    target_app = ops.gray_hist_matching(target_app_raw, style_app)

    # 4. Stylize
    print("[Demo] Stylizing...")
    start_stylize = time.time()
    output = fb_ti.stylize_with_guides(target_pos, target_app, patch_size=3)
    # Result is torch tensor on device
    output_np = output.cpu().numpy()
    print(f"[Demo] Taichi Stylization: {time.time() - start_stylize:.4f}s")

    # 5. Post-process (Skin blending)
    # FaceBlit uses a skin mask for blending with background
    skin_mask = ops.get_skin_mask(target_img, target_landmarks)
    final_output = ops.alpha_blend(output_np, target_img, skin_mask, sigma=25.0)

    out_file = output_dir / "target2_stylized_taichi.png"
    cv2.imwrite(str(out_file), final_output)
    print(f"[Demo] Success! Saved to {out_file}")


if __name__ == "__main__":
    main()
