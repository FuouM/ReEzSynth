from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.backends.commons import (
    alpha_blend,
    get_app_guide,
    get_head_area_rect,
    get_skin_mask,
    gradient_guide,
    gray_hist_matching,
    warp_mls_similarity,
)
from src.backends.taichi_backend import compute_lut_taichi, stylize_blit_taichi
from src.backends.torch_backend import (
    compute_look_up_cube_optimized,
    style_blit_voting_torch,
)
from src.backends.torch_backend_numba import (
    denoise_nnf_numba,
    initialize_nnf_dfs_voting_numba,
)
from src.backends.torch_backend_numpy import (
    _lookup_coords_from_guides,
    denoise_nnf_python,
    dfs_seed_grow_voting_numpy,
    initialize_nnf_vectorized,
    prepare_dfs_voting_arrays,
    voting_on_rgb,
)
from src.utils_face import detect_landmarks
from src.utils_io import (
    load_look_up_cube,
    pack_lut_packed,
    read_image_pil,
    read_landmarks_file,
    save_look_up_cube,
    to_numpy_image,
    to_tensor,
    to_tensor_simple,
    unpack_lut_packed,
    write_image_pil,
)


def save_landmarks_txt(landmarks, landmarks_path):
    with landmarks_path.open("w") as f:
        f.write("68\n")
        for x, y in landmarks:
            f.write(f"{x} {y}\n")


class FaceBlitInput_Style:
    def __init__(
        self,
        style_image,
        style_pos_guide,
        style_app_guide,
        style_landmarks,
        look_up_cube,
    ) -> None:
        # BGR UINT8
        self.style_image = style_image
        self.style_pos_guide = style_pos_guide
        # GRAYSCALE
        self.style_app_guide = style_app_guide
        # Clamped
        self.style_landmarks = style_landmarks

        self.look_up_cube = look_up_cube

    @classmethod
    def get_style_asset_paths(cls, out_dir, path_style_image):
        style_path = Path(path_style_image)
        stem = style_path.stem
        return {
            "path_style_pos_guide": out_dir / f"{stem}_style_pos.png",
            "path_style_app_guide": out_dir / f"{stem}_style_app.png",
            "path_style_landmarks": out_dir / f"{stem}_landmarks.txt",
            "path_style_lookup_table": out_dir / f"{stem}_lut.bytes",
        }

    @classmethod
    def compute_style_assets(
        cls,
        output_dir,
        path_style_image,
        landmark_model,
        predictor_path,
        device,
        lambda_pos: int = 10,
        lambda_app: int = 2,
        search_radius=30,
        draw_grid=False,
        stretch_hist=True,
        backend: Literal["taichi", "torch"] = "torch",
    ):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        asset_paths = FaceBlitInput_Style.get_style_asset_paths(
            out_dir, path_style_image
        )
        style_image = read_image_pil(path_style_image)

        landmarks = detect_landmarks(
            style_image, landmark_model, predictor_path, device
        )
        save_landmarks_txt(landmarks, asset_paths["path_style_landmarks"])

        style_pos_guide = gradient_guide(
            style_image.shape[1], style_image.shape[0], draw_grid=draw_grid
        )
        write_image_pil(asset_paths["path_style_pos_guide"], style_pos_guide)

        style_app_guide = get_app_guide(style_image, stretch_hist=stretch_hist)
        write_image_pil(asset_paths["path_style_app_guide"], style_app_guide)

        if backend == "taichi":
            style_lookup_packed = compute_lut_taichi(
                to_tensor_simple(style_pos_guide),
                to_tensor_simple(style_app_guide),
                lambda_pos=lambda_pos,
                lambda_app=lambda_app,
                search_radius=search_radius,
                device=device,
            )
            save_look_up_cube(
                unpack_lut_packed(style_lookup_packed),
                asset_paths["path_style_lookup_table"],
            )
        elif backend == "torch":
            style_lookup_table = compute_look_up_cube_optimized(
                style_pos_guide,
                style_app_guide,
                lambda_pos=lambda_pos,
                lambda_app=lambda_app,
                search_radius=search_radius,
                device=device,
            )
            save_look_up_cube(
                style_lookup_table, asset_paths["path_style_lookup_table"]
            )

        return asset_paths

    @classmethod
    def load_style_assets(
        cls,
        path_style_image,
        path_style_pos_guide,
        path_style_app_guide,
        path_style_landmarks,
        path_style_lookup_table,
    ):
        style_image = read_image_pil(path_style_image)
        style_pos_guide = read_image_pil(path_style_pos_guide)
        style_app_guide = read_image_pil(path_style_app_guide, convert="L")
        style_landmarks = read_landmarks_file(path_style_landmarks)
        look_up_cube = load_look_up_cube(path_style_lookup_table)
        return FaceBlitInput_Style(
            style_image, style_pos_guide, style_app_guide, style_landmarks, look_up_cube
        )


class FaceBlitInput_Target:
    def __init__(self, target_image, target_landmarks) -> None:
        self.target_image = target_image
        self.target_landmarks = target_landmarks


class FaceBlitEngine:
    def __init__(self):
        pass

    def stylize_image_with_guide_and_landmarks(
        self,
        input_target: FaceBlitInput_Target,
        input_style: FaceBlitInput_Style,
        device: str,
        dfs_mode: Literal["auto", "python", "numba"] = "auto",
        grid_size=3,
        patch_size=3,
        lambda_pos=10,
        lambda_app=2,
        use_vectorized=True,
        blend_sigma=25.0,
        stylize_background=False,
        backend: Literal["taichi", "torch"] = "torch",
        threshold=50,
        denoise_iters=1,
    ):
        mls_style_landmarks = list(input_style.style_landmarks)
        mls_target_landmarks = list(input_target.target_landmarks)
        if stylize_background:
            style_h, style_w = input_style.style_image.shape[:2]
            tgt_h, tgt_w = input_target.target_image.shape[:2]
            mls_style_landmarks.extend([(0, style_h), (style_w, style_h)])
            mls_target_landmarks.extend([(0, tgt_h), (tgt_w, tgt_h)])

        target_pos_guide = self.create_target_pos_guide(
            input_style.style_pos_guide,
            mls_style_landmarks,
            mls_target_landmarks,
            grid_size,
            device,
        )
        stylization_rect = get_head_area_rect(
            input_target.target_landmarks, input_target.target_image.shape[:2]
        )
        target_app_guide = get_app_guide(input_target.target_image, stretch_hist=False)
        target_app_matched = gray_hist_matching(
            target_app_guide, input_style.style_app_guide
        )

        if backend == "torch":
            # use_vectorized only affects style_blit_cpu (DFS vs LUT-vectorized NNF init).
            # style_blit_voting_torch always does GPU LUT lookup for the initial NNF.
            if device == "cpu" or not use_vectorized:
                stylized = style_blit_cpu(
                    input_style.style_pos_guide,
                    target_pos_guide,
                    input_style.style_app_guide,
                    target_app_matched,
                    input_style.look_up_cube,
                    input_style.style_image,
                    stylization_rect=stylization_rect,
                    patch_size=patch_size,
                    lambda_pos=lambda_pos,
                    lambda_app=lambda_app,
                    backend=dfs_mode,
                    use_vectorized=use_vectorized,
                    threshold=threshold,
                    denoise_iters=denoise_iters,
                )
            else:
                stylized = style_blit_voting_torch(
                    input_style.style_pos_guide,
                    target_pos_guide,
                    input_style.style_app_guide,
                    target_app_matched,
                    input_style.look_up_cube,
                    input_style.style_image,
                    stylization_rect,
                    patch_size,
                    lambda_pos,
                    lambda_app,
                    device,
                )
        elif backend == "taichi":
            stylized = stylize_blit_taichi(
                to_tensor_simple(input_style.style_image),
                to_tensor_simple(input_style.style_pos_guide),
                to_tensor_simple(input_style.style_app_guide),
                to_tensor_simple(target_pos_guide),
                to_tensor_simple(target_app_matched),
                pack_lut_packed(input_style.look_up_cube),
                device,
                stylization_rect=stylization_rect,
                patch_size=patch_size,
                lambda_pos=lambda_pos,
                lambda_app=lambda_app,
                threshold=threshold,
                use_vectorized=use_vectorized,
                denoise_iters=denoise_iters,
            )

        if not stylize_background:
            result = alpha_blend(
                stylized,
                input_target.target_image,
                get_skin_mask(input_target.target_image, input_target.target_landmarks),
                sigma=blend_sigma,
            )
        else:
            result = stylized

        return result, target_pos_guide

    def create_target_pos_guide(
        self, style_pos_guide, style_landmarks, target_landmarks, grid_size, device
    ):
        # MLS deformation of style position guide toward target landmarks
        pos_tensor = to_tensor(style_pos_guide).to(device)
        warped = warp_mls_similarity(
            pos_tensor,
            torch.tensor(style_landmarks, device=device, dtype=torch.float32),
            torch.tensor(target_landmarks, device=device, dtype=torch.float32),
            grid_size=grid_size,
        )
        target_pos_guide = to_numpy_image(warped)
        return target_pos_guide


def style_blit_cpu(
    style_pos_guide,
    target_pos_guide,
    style_app_guide,
    target_app_guide,
    look_up_cube,
    style_image,
    stylization_rect,
    patch_size,
    lambda_pos,
    lambda_app,
    backend: Literal["auto", "python", "numba"],
    use_vectorized=True,
    threshold=None,
    denoise_iters=1,
):
    h_t, w_t = target_pos_guide.shape[:2]
    x, y, w, h = stylization_rect
    style_h, style_w = style_image.shape[:2]
    target_app_gray = target_app_guide

    if use_vectorized:
        nnf = initialize_nnf_vectorized(
            target_pos_guide,
            target_app_gray,
            look_up_cube,
            (style_h, style_w),
            (x, y, w, h),
        )
        if patch_size > 1:
            for _ in range(max(0, int(denoise_iters))):
                nnf = _denoise_nnf_cpu(nnf, patch_size, backend)
    else:
        nnf = initialize_nnf(
            target_pos_guide,
            target_app_gray,
            style_pos_guide,
            style_app_guide,
            threshold,
            look_up_cube,
            lambda_pos,
            lambda_app,
            style_h,
            style_w,
            x,
            y,
            w,
            h,
            h_t,
            w_t,
            backend=backend,
        )
        nnf = _denoise_nnf_cpu(nnf, patch_size, backend)

    output = voting_on_rgb(style_image, nnf, patch_size)
    return output


def _denoise_nnf_cpu(
    nnf: np.ndarray, patch_size: int, backend: Literal["auto", "python", "numba"]
) -> np.ndarray:
    if backend == "python":
        return denoise_nnf_python(nnf, patch_size)
    if backend in ("auto", "numba"):
        return denoise_nnf_numba(nnf, patch_size)
    raise ValueError(f"backend must be 'auto', 'python', or 'numba', got {backend!r}")


def initialize_nnf(
    target_pos_guide,
    target_app_gray,
    style_pos_guide,
    style_app_guide,
    threshold,
    look_up_cube,
    lambda_pos,
    lambda_app,
    style_h,
    style_w,
    x,
    y,
    w,
    h,
    h_t,
    w_t,
    backend: Literal["auto", "python", "numba"],
):
    """DFS-based region growing NNF, then :func:`denoise_nnf` (pure Python / NumPy)."""
    if backend == "python":
        covered_pixels, seeds = prepare_dfs_voting_arrays(
            target_pos_guide,
            target_app_gray,
            h_t,
            w_t,
            x,
            y,
            h,
            w,
            look_up_cube,
            style_h,
            style_w,
            lambda_app,
        )
        nnf = np.zeros((h_t, w_t, 2), dtype=np.int32)

        chunk_number = 1
        for row_t, col_t, style_seed_point in seeds:
            if covered_pixels[row_t, col_t] != 0:
                continue
            target_seed_point = (row_t, col_t)
            dfs_seed_grow_voting_numpy(
                target_seed_point,
                style_seed_point,
                style_pos_guide,
                target_pos_guide,
                style_app_guide,
                target_app_gray,
                nnf,
                covered_pixels,
                chunk_number,
                threshold,
                lambda_pos,
                lambda_app,
            )
            chunk_number += 1
    else:
        nnf, covered_pixels = initialize_nnf_dfs_voting_numba(
            target_pos_guide,
            target_app_gray,
            style_pos_guide,
            style_app_guide,
            look_up_cube,
            threshold,
            lambda_pos,
            lambda_app,
            style_h,
            style_w,
            x,
            y,
            h,
            w,
            h_t,
            w_t,
        )

    # DFS only seeds the head box and propagates with a fixed style–target
    # translation. Pixels that go out of style bounds or fail the error
    # threshold are never visited; without a fill, they stay (0,0) and
    # patch-voting leaves them black. Use the same LUT as vectorized init.
    hole = covered_pixels == 0
    if np.any(hole):
        cx, cy = _lookup_coords_from_guides(
            target_pos_guide,
            target_app_gray,
            look_up_cube,
            (style_h, style_w),
        )
        nnf[..., 0][hole] = cy[hole]
        nnf[..., 1][hole] = cx[hole]

    return nnf
