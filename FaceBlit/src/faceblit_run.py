# ruff: noqa: E402

import sys
import time
from pathlib import Path
from typing import Literal

import cv2

# Add project roots to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FaceBlit"))

from src.backends.taichi_backend import ensure_ti_init
from src.engine import (
    FaceBlitEngine,
    FaceBlitInput_Style,
    FaceBlitInput_Target,
)
from src.utils_face import detect_landmarks

MODEL_DIR = ROOT / "FaceBlit" / "models"
PREDICTOR_PATH = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"


def run_faceblit(
    style_path,
    target_path,
    output_dir,
    backend: Literal["torch", "taichi"],
    landmark_backend: Literal["dlib", "fan"],
    device,
    stylize_background=False,
    dfs_mode="auto",
    use_vectorized=False,
):
    predictor = PREDICTOR_PATH if landmark_backend == "dlib" else None

    if backend == "taichi":
        ensure_ti_init(device)

    print(f"[Demo] {backend} FaceBlit")
    t0 = time.time()
    faceblit_engine = FaceBlitEngine()

    # Target input
    target_img = cv2.imread(target_path)
    target_stem = Path(target_path).stem
    target_landmarks = detect_landmarks(target_img, landmark_backend, predictor, device)
    input_target = FaceBlitInput_Target(target_img, target_landmarks)

    # Style input

    asset_paths = FaceBlitInput_Style.compute_style_assets(
        output_dir,
        style_path,
        landmark_backend,
        predictor,
        device,
        draw_grid=False,
        stretch_hist=True,
        backend=backend,
    )

    input_style = FaceBlitInput_Style.load_style_assets(
        style_path,
        asset_paths["path_style_pos_guide"],
        asset_paths["path_style_app_guide"],
        asset_paths["path_style_landmarks"],
        asset_paths["path_style_lookup_table"],
    )

    result, target_pos_guide = faceblit_engine.stylize_image_with_guide_and_landmarks(
        input_target=input_target,
        input_style=input_style,
        device=device,
        grid_size=3,
        patch_size=3,
        lambda_pos=10,
        lambda_app=2,
        use_vectorized=use_vectorized,
        dfs_mode=dfs_mode,
        blend_sigma=25,
        stylize_background=stylize_background,
        backend=backend,
    )

    print(f"[Demo] {backend}: stylization: {time.time() - t0:.4f}s")

    out_file = output_dir / f"{target_stem}_stylized_{backend}.png"
    cv2.imwrite(str(out_file), result)

    pos_out = output_dir / f"{target_stem}_target_pos_{backend}.png"
    cv2.imwrite(str(pos_out), target_pos_guide)

    print(f"[Demo] Success! Saved to {out_file} (pos guide: {pos_out})")


if __name__ == "__main__":
    examples_dir = ROOT / "FaceBlit" / "examples"
    # taichi_output_dir = ROOT / "FaceBlit" / "test_output" / "taichi_cpu_novec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=taichi_output_dir,
    #     backend="taichi",
    #     landmark_backend="dlib",
    #     device="cpu",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=False,
    # )
    # taichi_output_dir = ROOT / "FaceBlit" / "test_output" / "taichi_cpu_vec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=taichi_output_dir,
    #     backend="taichi",
    #     landmark_backend="dlib",
    #     device="cpu",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=True,
    # )
    # taichi_output_dir = ROOT / "FaceBlit" / "test_output" / "taichi_gpu_novec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=taichi_output_dir,
    #     backend="taichi",
    #     landmark_backend="dlib",
    #     device="cuda",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=False,
    # )
    # taichi_output_dir = ROOT / "FaceBlit" / "test_output" / "taichi_gpu_vec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=taichi_output_dir,
    #     backend="taichi",
    #     landmark_backend="dlib",
    #     device="cuda",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=True,
    # )

    # torch_output_dir = ROOT / "FaceBlit" / "test_output" / "torch_cpu_novec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=torch_output_dir,
    #     backend="torch",
    #     landmark_backend="dlib",
    #     device="cpu",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=False,
    # )
    # torch_output_dir = ROOT / "FaceBlit" / "test_output" / "torch_cpu_vec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=torch_output_dir,
    #     backend="torch",
    #     landmark_backend="dlib",
    #     device="cpu",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=True,
    # )
    # torch_output_dir = ROOT / "FaceBlit" / "test_output" / "torch_gpu_novec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=torch_output_dir,
    #     backend="torch",
    #     landmark_backend="dlib",
    #     device="cuda",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=False,
    # )
    # torch_output_dir = ROOT / "FaceBlit" / "test_output" / "torch_gpu_vec"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=torch_output_dir,
    #     backend="torch",
    #     landmark_backend="dlib",
    #     device="cuda",
    #     stylize_background=False,
    #     dfs_mode="auto",
    #     use_vectorized=True,
    # )
    # taichi_output_dir = ROOT / "FaceBlit" / "test_output" / "taichi_gpu_novec_bg"
    # run_faceblit(
    #     style_path=examples_dir / "style_watercolorgirl.png",
    #     target_path=examples_dir / "target2.png",
    #     output_dir=taichi_output_dir,
    #     backend="taichi",
    #     landmark_backend="dlib",
    #     device="cuda",
    #     stylize_background=True,
    #     dfs_mode="auto",
    #     use_vectorized=False,
    # )
    torch_output_dir = ROOT / "FaceBlit" / "test_output" / "torch_gpu_novec_bg"
    run_faceblit(
        style_path=examples_dir / "style_watercolorgirl.png",
        target_path=examples_dir / "target2.png",
        output_dir=torch_output_dir,
        backend="torch",
        landmark_backend="dlib",
        device="cuda",
        stylize_background=True,
        dfs_mode="auto",
        use_vectorized=False,
    )
