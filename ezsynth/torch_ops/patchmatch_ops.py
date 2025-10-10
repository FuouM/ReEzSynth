# ezsynth/torch_ops/patchmatch_ops.py
"""
PatchMatch algorithm operations for EBSynth.

This module implements the core PatchMatch operations:
- try_patch_batch: Evaluate candidate patches and update NNF
- propagation_step: Spatial coherence via neighbor propagation
- random_search_step: Exploration via random offsets
"""

import os
from typing import Tuple

import torch

from .omega_ops import compute_omega_scores, update_omega_map
from .patch_ops import compute_patch_ncc_vectorized, compute_patch_ssd_vectorized

skip_metal = os.environ.get("EZSYNTH_SKIP_METAL", "").lower() in ("1", "true", "yes")
skip_metal_verbose = os.environ.get("EZSYNTH_SKIP_METAL_VERBOSE", "").lower() in (
    "1",
    "true",
    "yes",
)

# Try to import Metal operations (optional)
if not skip_metal:
    try:
        from .metal_ext.compiler import compiled_metal_ops

        METAL_AVAILABLE = True
        if skip_metal_verbose:
            print("Loaded custom Metal operations for patchmatch_ops.")
    except ImportError:
        METAL_AVAILABLE = False
        compiled_metal_ops = None
        if skip_metal_verbose:
            print("Failed to import custom Metal operations for patchmatch_ops.")
else:
    METAL_AVAILABLE = False
    compiled_metal_ops = None
    if skip_metal_verbose:
        print(
            "Skipping custom Metal operations for patchmatch_ops (EZSYNTH_SKIP_METAL set)."
        )

# Cost function constants
COST_FUNCTION_SSD = 0
COST_FUNCTION_NCC = 1


def try_patch_batch(
    candidate_coords: torch.Tensor,
    current_nnf: torch.Tensor,
    current_errors: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    cost_function_mode: int,
    omega_best: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try candidate patches and update NNF if better.
    """
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2
    valid_candidates = (
        (candidate_coords[..., 0] >= r)
        & (candidate_coords[..., 0] < W_s - r)
        & (candidate_coords[..., 1] >= r)
        & (candidate_coords[..., 1] < H_s - r)
    )

    if cost_function_mode == COST_FUNCTION_NCC:
        candidate_patch_errors = compute_patch_ncc_vectorized(
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            candidate_coords,
            patch_size,
            style_weights,
            guide_weights,
        )
    else:  # SSD
        candidate_patch_errors = compute_patch_ssd_vectorized(
            source_style_patches, target_style_patches, candidate_coords, style_weights
        )
        if source_guide_patches.numel() > 0:
            guide_errors = compute_patch_ssd_vectorized(
                source_guide_patches,
                target_guide_patches,
                candidate_coords,
                guide_weights,
            )
            candidate_patch_errors += guide_errors

    candidate_omega_scores = compute_omega_scores(
        omega_map, candidate_coords, patch_size
    )
    current_omega_scores = compute_omega_scores(omega_map, current_nnf, patch_size)
    patch_pixel_count = patch_size * patch_size
    candidate_omega_normalized = candidate_omega_scores / (
        patch_pixel_count * omega_best
    )
    current_omega_normalized = current_omega_scores / (patch_pixel_count * omega_best)
    candidate_total_error = (
        candidate_patch_errors + uniformity_weight * candidate_omega_normalized
    )
    current_total_error = current_errors + uniformity_weight * current_omega_normalized
    update_mask = (candidate_total_error < current_total_error) & valid_candidates
    updated_nnf = torch.where(
        update_mask.unsqueeze(2).expand_as(current_nnf), candidate_coords, current_nnf
    )
    updated_errors = torch.where(update_mask, candidate_patch_errors, current_errors)
    return updated_nnf, updated_errors, update_mask


def propagation_step(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    is_odd: bool,
    mask: torch.Tensor,
    cost_function_mode: int,
    omega_best: float,
):
    """
    Propagation step: try neighbors' matches for spatial coherence.
    """
    H_s, W_s = source_style_patches.shape[:2]
    device = nnf.device
    active_mask = mask == 255

    # Determine neighbor offsets based on iteration parity
    x_offset, y_offset = (-1, 0) if is_odd else (1, 0)

    # --- Horizontal Propagation ---
    neighbor_nnf = torch.roll(nnf, shifts=x_offset, dims=1)
    if x_offset == -1:  # Pad right
        neighbor_nnf[:, -1, :] = neighbor_nnf[:, -2, :]
    else:  # Pad left
        neighbor_nnf[:, 0, :] = neighbor_nnf[:, 1, :]

    horiz_candidates = neighbor_nnf - torch.tensor([x_offset, 0], device=device)

    new_nnf, new_errors, horiz_updates = try_patch_batch(
        horiz_candidates,
        nnf,
        error_map,
        omega_map,
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        style_weights,
        guide_weights,
        uniformity_weight,
        patch_size,
        cost_function_mode,
        omega_best,
    )

    changed_indices = (horiz_updates & active_mask).nonzero(as_tuple=False)
    if len(changed_indices) > 0:
        y_c, x_c = changed_indices[:, 0], changed_indices[:, 1]
        update_omega_map(omega_map, nnf[y_c, x_c], new_nnf[y_c, x_c], patch_size)
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)

    # --- Vertical Propagation ---
    y_offset = -1 if is_odd else 1
    neighbor_nnf = torch.roll(nnf, shifts=y_offset, dims=0)
    if y_offset == -1:  # Pad bottom
        neighbor_nnf[-1, :, :] = neighbor_nnf[-2, :, :]
    else:  # Pad top
        neighbor_nnf[0, :, :] = neighbor_nnf[1, :, :]

    vert_candidates = neighbor_nnf - torch.tensor([0, y_offset], device=device)

    new_nnf, new_errors, vert_updates = try_patch_batch(
        vert_candidates,
        nnf,
        error_map,
        omega_map,
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        style_weights,
        guide_weights,
        uniformity_weight,
        patch_size,
        cost_function_mode,
        omega_best,
    )

    changed_indices = (vert_updates & active_mask).nonzero(as_tuple=False)
    if len(changed_indices) > 0:
        y_c, x_c = changed_indices[:, 0], changed_indices[:, 1]
        update_omega_map(omega_map, nnf[y_c, x_c], new_nnf[y_c, x_c], patch_size)
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)


def random_search_step(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    initial_radius: int,
    mask: torch.Tensor,
    search_pruning_threshold: float,
    cost_function_mode: int,
    omega_best: float,
    generator: torch.Generator = None,
):
    """
    Optimized random search with pruning.
    """
    device = nnf.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2

    # --- 1. Pruning: Select only active pixels ---
    active_mask = mask == 255
    if search_pruning_threshold > 0:
        active_mask &= error_map >= search_pruning_threshold

    active_indices = active_mask.nonzero(as_tuple=True)
    if active_indices[0].numel() == 0:
        return  # Exit early if no pixels to process

    y_coords, x_coords = active_indices
    num_active = y_coords.numel()

    radius = initial_radius
    while radius >= 1:
        # --- 2. Generate candidates only for active pixels ---
        current_nnf_active = nnf[y_coords, x_coords]

        rand_offsets_x = torch.randint(
            -radius, radius + 1, (num_active,), device=device, generator=generator
        )
        rand_offsets_y = torch.randint(
            -radius, radius + 1, (num_active,), device=device, generator=generator
        )

        candidate_coords_active = current_nnf_active + torch.stack(
            [rand_offsets_x, rand_offsets_y], dim=-1
        )

        # --- 3. Validate active candidates ---
        valid_mask_active = (
            (candidate_coords_active[..., 0] >= r)
            & (candidate_coords_active[..., 0] < W_s - r)
            & (candidate_coords_active[..., 1] >= r)
            & (candidate_coords_active[..., 1] < H_s - r)
        )

        valid_indices_in_active = valid_mask_active.nonzero(as_tuple=True)[0]
        if valid_indices_in_active.numel() == 0:
            radius //= 2
            continue

        # Filter all data down to only the valid candidates to process
        y_v = y_coords[valid_indices_in_active]
        x_v = x_coords[valid_indices_in_active]
        cand_v = candidate_coords_active[valid_indices_in_active]
        curr_nnf_v = current_nnf_active[valid_indices_in_active]

        # --- 4. Compute errors only for the valid active candidates ---
        # Our modified cost functions can handle this flattened data directly
        target_style_p_v = target_style_patches[y_v, x_v]
        target_guide_p_v = (
            target_guide_patches[y_v, x_v]
            if target_guide_patches.numel() > 0
            else target_guide_patches
        )

        if cost_function_mode == COST_FUNCTION_NCC:
            cand_errors_v = compute_patch_ncc_vectorized(
                source_style_patches,
                target_style_p_v,
                source_guide_patches,
                target_guide_p_v,
                cand_v,
                patch_size,
                style_weights,
                guide_weights,
            )
        else:  # SSD
            cand_errors_v = compute_patch_ssd_vectorized(
                source_style_patches, target_style_p_v, cand_v, style_weights
            )
            if source_guide_patches.numel() > 0:
                cand_errors_v += compute_patch_ssd_vectorized(
                    source_guide_patches, target_guide_p_v, cand_v, guide_weights
                )

        # --- 5. Compare total error and update ---
        patch_pixel_count = patch_size * patch_size

        cand_omega_v = compute_omega_scores(
            omega_map, cand_v.unsqueeze(0), patch_size
        ).squeeze()
        curr_omega_v = compute_omega_scores(
            omega_map, curr_nnf_v.unsqueeze(0), patch_size
        ).squeeze()

        cand_total_error = cand_errors_v + uniformity_weight * (
            cand_omega_v / (patch_pixel_count * omega_best)
        )
        curr_total_error = error_map[y_v, x_v] + uniformity_weight * (
            curr_omega_v / (patch_pixel_count * omega_best)
        )

        update_mask_v = cand_total_error < curr_total_error

        update_indices_in_v = update_mask_v.nonzero(as_tuple=True)[0]
        if update_indices_in_v.numel() > 0:
            # Get original coordinates of pixels to update
            final_y = y_v[update_indices_in_v]
            final_x = x_v[update_indices_in_v]

            old_nnf_vals = nnf[final_y, final_x]
            new_nnf_vals = cand_v[update_indices_in_v]

            update_omega_map(omega_map, old_nnf_vals, new_nnf_vals, patch_size)
            nnf[final_y, final_x] = new_nnf_vals.to(nnf.dtype)

            error_map[final_y, final_x] = cand_errors_v[update_indices_in_v]

        radius //= 2
