# ezsynth/torch_ops/patchmatch_ops.py
"""
PatchMatch algorithm operations for EBSynth.

This module implements the core PatchMatch operations:
- try_patch_batch: Evaluate candidate patches and update NNF
- propagation_step: Spatial coherence via neighbor propagation
- random_search_step: Exploration via random offsets
"""

from typing import Tuple

import torch

from .omega_ops import compute_omega_scores, update_omega_map
from .patch_ops import compute_patch_ncc_vectorized, compute_patch_ssd_vectorized

# Cost function constants
COST_FUNCTION_SSD = 0
COST_FUNCTION_NCC = 1

# OPTIMIZATION: Cache for small offset tensors to avoid repeated allocations
_OFFSET_TENSOR_CACHE = {}


def _get_offset_tensor(offset_tuple, device):
    """Get or create cached offset tensor."""
    key = (offset_tuple, str(device))
    if key not in _OFFSET_TENSOR_CACHE:
        _OFFSET_TENSOR_CACHE[key] = torch.tensor(list(offset_tuple), device=device)
    return _OFFSET_TENSOR_CACHE[key]


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
    source_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    target_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    update_nnf: bool = True,
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
            source_stats,
            target_stats,
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
    if update_nnf:
        updated_nnf = torch.where(
            update_mask.unsqueeze(2).expand_as(current_nnf),
            candidate_coords,
            current_nnf,
        )
        updated_errors = torch.where(
            update_mask, candidate_patch_errors, current_errors
        )
    else:
        # For initial error computation, don't update NNF, just compute errors
        updated_nnf = current_nnf
        updated_errors = candidate_patch_errors  # Use the computed errors
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
    source_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    target_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
):
    """
    Propagation step: try neighbors' matches for spatial coherence.
    OPTIMIZED: Batches omega map updates to reduce overhead.

    Corrected Logic to match CUDA:
    - is_odd=True  => Forward Pass (Iter 1, 3...): Look Left/Top, Propagate +1
    - is_odd=False => Backward Pass (Iter 0, 2...): Look Right/Bottom, Propagate -1
    """
    H_s, W_s = source_style_patches.shape[:2]
    device = nnf.device
    active_mask = mask == 255

    # OPTIMIZATION: Store original NNF before any updates for omega tracking
    nnf_original = nnf.clone()

    # Determine propagation direction and neighbor lookup
    if is_odd:
        # Forward Pass: Look Left (-1, 0) and Top (0, -1)
        # Propagate: Neighbor + 1
        # To read Left neighbor at index i, we need to shift Right (+1)
        shift_x, shift_y = 1, 1
        prop_x, prop_y = 1, 1
    else:
        # Backward Pass: Look Right (+1, 0) and Bottom (0, +1)
        # Propagate: Neighbor - 1
        # To read Right neighbor at index i, we need to shift Left (-1)
        shift_x, shift_y = -1, -1
        prop_x, prop_y = -1, -1

    # --- Horizontal Propagation ---
    # Roll to align neighbor with current pixel
    neighbor_nnf = torch.roll(nnf, shifts=shift_x, dims=1)

    # Handle boundary conditions (padding)
    if shift_x == 1:  # Reading from Left
        # Column 0 wraps from N-1. Replace with self (no propagation from left edge)
        neighbor_nnf[:, 0, :] = nnf[:, 0, :]
    else:  # Reading from Right
        # Column N-1 wraps from 0. Replace with self
        neighbor_nnf[:, -1, :] = nnf[:, -1, :]

    # Apply propagation offset
    horiz_candidates = neighbor_nnf + _get_offset_tensor((prop_x, 0), device)

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
        source_stats,
        target_stats,
    )

    # OPTIMIZATION: Collect changes but don't update omega yet
    horiz_changed_indices = (horiz_updates & active_mask).nonzero(as_tuple=False)

    # Update NNF and errors for vertical propagation
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)

    # --- Vertical Propagation ---
    neighbor_nnf = torch.roll(nnf, shifts=shift_y, dims=0)

    if shift_y == 1:  # Reading from Top
        neighbor_nnf[0, :, :] = nnf[0, :, :]
    else:  # Reading from Bottom
        neighbor_nnf[-1, :, :] = nnf[-1, :, :]

    vert_candidates = neighbor_nnf + _get_offset_tensor((0, prop_y), device)

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
        source_stats,
        target_stats,
    )

    # OPTIMIZATION: Collect vertical changes
    vert_changed_indices = (vert_updates & active_mask).nonzero(as_tuple=False)

    # Update NNF and errors
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)

    # Apply Horizontal Updates to Omega
    if len(horiz_changed_indices) > 0:
        y_h, x_h = horiz_changed_indices[:, 0], horiz_changed_indices[:, 1]
        update_omega_map(
            omega_map, nnf_original[y_h, x_h], nnf[y_h, x_h], patch_size
        )  # nnf here is intermediate (post-horiz)

        # Update nnf_original to match current nnf for the next step
        # This is needed so vertical updates use the correct "old" value
        nnf_original[y_h, x_h] = nnf[y_h, x_h]

    # Apply Vertical Updates to Omega
    if len(vert_changed_indices) > 0:
        y_v, x_v = vert_changed_indices[:, 0], vert_changed_indices[:, 1]
        # nnf_original now holds the state before vertical update (because we updated it above)
        # nnf now holds the state after vertical update
        update_omega_map(omega_map, nnf_original[y_v, x_v], nnf[y_v, x_v], patch_size)


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
    source_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    target_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
):
    """
    Optimized random search with pruning.
    OPTIMIZED: Reduced omega computations and pre-computed constants.
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

    # OPTIMIZATION: Pre-compute constants outside the loop
    patch_pixel_count = patch_size * patch_size
    omega_normalization = patch_pixel_count * omega_best
    uniformity_factor = uniformity_weight / omega_normalization

    # OPTIMIZATION: Pre-extract current NNF for active pixels once
    current_nnf_active = nnf[y_coords, x_coords]

    # OPTIMIZATION: Pre-compute current omega scores once (reused across radius levels)
    curr_omega_active = compute_omega_scores(
        omega_map, current_nnf_active.unsqueeze(0), patch_size
    ).squeeze()

    radius = initial_radius
    while radius >= 1:
        # --- 2. Generate candidates only for active pixels ---
        if generator is not None:
            rand_offsets_x = torch.randint(
                -radius, radius + 1, (num_active,), device=device, generator=generator
            )
            rand_offsets_y = torch.randint(
                -radius, radius + 1, (num_active,), device=device, generator=generator
            )
        else:
            rand_offsets_x = torch.randint(
                -radius, radius + 1, (num_active,), device=device
            )
            rand_offsets_y = torch.randint(
                -radius, radius + 1, (num_active,), device=device
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

        # --- 4. Compute errors only for the valid active candidates ---
        target_style_p_v = target_style_patches[y_v, x_v]
        target_guide_p_v = (
            target_guide_patches[y_v, x_v]
            if target_guide_patches.numel() > 0
            else target_guide_patches
        )

        if cost_function_mode == COST_FUNCTION_NCC:
            # Slice target stats if available
            target_stats_v = None
            if target_stats is not None:
                t_vals, t_mean, t_std = target_stats
                target_stats_v = (t_vals[y_v, x_v], t_mean[y_v, x_v], t_std[y_v, x_v])

            cand_errors_v = compute_patch_ncc_vectorized(
                source_style_patches,
                target_style_p_v,
                source_guide_patches,
                target_guide_p_v,
                cand_v,
                patch_size,
                style_weights,
                guide_weights,
                source_stats,
                target_stats_v,
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
        # OPTIMIZATION: Only compute omega for candidates, reuse pre-computed current omega
        cand_omega_v = compute_omega_scores(
            omega_map, cand_v.unsqueeze(0), patch_size
        ).squeeze()

        # OPTIMIZATION: Use pre-computed current omega (indexed from pre-computed active omega)
        curr_omega_v = curr_omega_active[valid_indices_in_active]

        # OPTIMIZATION: Use pre-computed uniformity factor
        cand_total_error = cand_errors_v + uniformity_factor * cand_omega_v
        curr_total_error = error_map[y_v, x_v] + uniformity_factor * curr_omega_v

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

            # OPTIMIZATION: Update cached current NNF and omega for updated pixels
            # Map back from valid indices to active indices
            active_update_indices = valid_indices_in_active[update_indices_in_v]
            current_nnf_active[active_update_indices] = new_nnf_vals.to(
                current_nnf_active.dtype
            )
            curr_omega_active[active_update_indices] = cand_omega_v[update_indices_in_v]

        radius //= 2
