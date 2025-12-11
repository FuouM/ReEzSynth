"""
Core FaceBlit operations split into smaller, focused modules.
Public API remains compatible; functions are re-exported here.
"""

from .geometry import clamp_landmarks, get_head_area_rect
from .guides import (
    _gaussian_pyr_down,
    get_app_guide,
    gradient_guide,
    gray_hist_matching,
)
from .image_utils import _ensure_grayscale, _to_numpy_image, _to_tensor, _to_uint8
from .io_utils import read_landmarks_file
from .lookup import (
    _build_seed_maps,
    _distance_transform_with_indices,
    compute_look_up_cube,
    compute_look_up_cube_optimized,
    load_look_up_cube,
    save_look_up_cube,
)
from .masking import (
    _bgr_to_yuv_torch,
    _draw_ellipse_torch,
    _fill_convex_poly_torch,
    alpha_blend,
    get_skin_mask,
)
from .stylization import (
    _apply_rect_mask,
    _compute_style_seed_point,
    _denoise_nnf,
    _denoise_nnf_vectorized,
    _dfs_seed_grow_voting,
    _initialize_nnf_vectorized,
    _lookup_coords_from_guides,
    _pixel_out_of_range,
    _voting_on_rgb,
    compute_guided_error,
    dfs_seed_grow,
    style_blit,
    style_blit_voting,
)
from .warping import _calc_mls_delta, warp_mls_similarity

__all__ = [
    "_to_uint8",
    "_ensure_grayscale",
    "_to_tensor",
    "_to_numpy_image",
    "gradient_guide",
    "_gaussian_pyr_down",
    "get_app_guide",
    "gray_hist_matching",
    "_calc_mls_delta",
    "warp_mls_similarity",
    "_build_seed_maps",
    "_distance_transform_with_indices",
    "compute_look_up_cube",
    "compute_look_up_cube_optimized",
    "save_look_up_cube",
    "load_look_up_cube",
    "compute_guided_error",
    "dfs_seed_grow",
    "_pixel_out_of_range",
    "_compute_style_seed_point",
    "_dfs_seed_grow_voting",
    "_initialize_nnf_vectorized",
    "_denoise_nnf_vectorized",
    "_denoise_nnf",
    "_voting_on_rgb",
    "_lookup_coords_from_guides",
    "_apply_rect_mask",
    "style_blit",
    "style_blit_voting",
    "_bgr_to_yuv_torch",
    "_fill_convex_poly_torch",
    "_draw_ellipse_torch",
    "get_skin_mask",
    "alpha_blend",
    "clamp_landmarks",
    "get_head_area_rect",
    "read_landmarks_file",
]
