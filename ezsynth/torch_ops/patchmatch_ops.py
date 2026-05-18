# ezsynth/torch_ops/patchmatch_ops.py
"""
PatchMatch algorithm operations for EBSynth.

This module implements the core PatchMatch operations:
- try_patch_batch: Evaluate candidate patches and update NNF
- propagation_step: Spatial coherence via neighbor propagation
- random_search_step: Exploration via random offsets
"""

from typing import Optional, Tuple

import torch

from ezsynth.consts import COST_FUNCTION_NCC

from .device_cache import clear_torch_device_cache
from .microprofile import region as _mp_region
from .omega_ops import (
    gather_omega_scores_from_patches,
    gather_omega_scores_pair,
    omega_mean_map_buffer,
    omega_patch_mean_map_into,
    update_omega_map,
)
from .patch_ops import (
    compute_patch_ncc_vectorized,
    compute_patch_ssd_style_guide_fused,
    compute_patch_ssd_vectorized,
)

# OPTIMIZATION: Cache for small offset tensors to avoid repeated allocations
_OFFSET_TENSOR_CACHE = {}

# Reused snapshot for propagation omega deltas (avoids ``nnf.clone()`` each step).
_PROP_NNF_SNAPSHOT: dict = {}

# Reused random-search buffers keyed by device / NNF dtype. The active set is
# usually full-frame, so this removes repeated temporary allocations per radius.
_RANDOM_SEARCH_BUFFER_CACHE: dict = {}

# Reused neighbor shifted NNF for propagation (``torch.cat(..., out=buf)``).
_PROP_NEIGHBOR_NNF_BUF: dict = {}


def _prop_neighbor_nnf_buf(nnf: torch.Tensor) -> torch.Tensor:
    """Cached buffer for shifted neighbor ``nnf`` (same layout as ``nnf``)."""
    key = (nnf.shape, str(nnf.device), nnf.dtype)
    buf = _PROP_NEIGHBOR_NNF_BUF.get(key)
    if buf is None or buf.shape != nnf.shape:
        buf = torch.empty_like(nnf)
        _PROP_NEIGHBOR_NNF_BUF[key] = buf
    return buf


def _prop_nnf_snapshot(nnf: torch.Tensor) -> torch.Tensor:
    key = (nnf.shape, str(nnf.device), nnf.dtype)
    buf = _PROP_NNF_SNAPSHOT.get(key)
    if buf is None:
        buf = torch.empty_like(nnf)
        _PROP_NNF_SNAPSHOT[key] = buf
    buf.copy_(nnf)
    return buf


def _get_offset_tensor(offset_tuple, device):
    """Get or create cached offset tensor."""
    key = (offset_tuple, str(device))
    if key not in _OFFSET_TENSOR_CACHE:
        _OFFSET_TENSOR_CACHE[key] = torch.tensor(list(offset_tuple), device=device)
    return _OFFSET_TENSOR_CACHE[key]


def _random_search_buffers(
    num_active: int, nnf: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (str(nnf.device), nnf.dtype)
    entry = _RANDOM_SEARCH_BUFFER_CACHE.get(key)
    if entry is None or entry[0].shape[0] < num_active:
        rand_x = torch.empty((num_active,), dtype=nnf.dtype, device=nnf.device)
        rand_y = torch.empty((num_active,), dtype=nnf.dtype, device=nnf.device)
        candidate = torch.empty((num_active, 2), dtype=nnf.dtype, device=nnf.device)
        entry = (rand_x, rand_y, candidate)
        _RANDOM_SEARCH_BUFFER_CACHE[key] = entry
    rand_x, rand_y, candidate = entry
    return rand_x[:num_active], rand_y[:num_active], candidate[:num_active]


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
    target_modulation_patches: Optional[torch.Tensor] = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
    update_nnf: bool = True,
    *,
    omega_mean_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try candidate patches and update NNF if better.
    """
    _dev = candidate_coords.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2
    valid_candidates = (
        (candidate_coords[..., 0] >= r)
        & (candidate_coords[..., 0] < W_s - r)
        & (candidate_coords[..., 1] >= r)
        & (candidate_coords[..., 1] < H_s - r)
    )

    if cost_function_mode == COST_FUNCTION_NCC:
        with _mp_region("try_patch:ncc", _dev):
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
                target_modulation_patches,
                use_bilateral=use_bilateral,
                sigma_spatial=sigma_spatial,
                sigma_color=sigma_color,
                n_size_step=n_size_step,
            )
    else:  # SSD
        ssd_kw = dict(
            use_bilateral=use_bilateral,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            n_size_step=n_size_step,
        )
        if (
            source_guide_patches.numel() > 0
            and (not use_bilateral)
            and n_size_step == 1
        ):
            with _mp_region("try_patch:ssd_fused", _dev):
                candidate_patch_errors = compute_patch_ssd_style_guide_fused(
                    source_style_patches,
                    target_style_patches,
                    source_guide_patches,
                    target_guide_patches,
                    candidate_coords,
                    style_weights,
                    guide_weights,
                    target_modulation_patches,
                    **ssd_kw,
                )
        else:
            with _mp_region("try_patch:ssd_style", _dev):
                candidate_patch_errors = compute_patch_ssd_vectorized(
                    source_style_patches,
                    target_style_patches,
                    candidate_coords,
                    style_weights,
                    None,
                    **ssd_kw,
                )
            if source_guide_patches.numel() > 0:
                with _mp_region("try_patch:ssd_guide", _dev):
                    candidate_patch_errors = (
                        candidate_patch_errors
                        + compute_patch_ssd_vectorized(
                            source_guide_patches,
                            target_guide_patches,
                            candidate_coords,
                            guide_weights,
                            target_modulation_patches,
                            source_style_patches_for_bilateral=source_style_patches,
                            **ssd_kw,
                        )
                    )

    if uniformity_weight == 0.0:
        with _mp_region("try_patch:omega_gather_and_cost", _dev):
            update_mask = (candidate_patch_errors < current_errors) & valid_candidates
    else:
        if omega_mean_map is None:
            with _mp_region("try_patch:omega_mean_build", _dev):
                omega_mean_buf = omega_mean_map_buffer(_dev, H_s, W_s)
                omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
                omega_mean_map = omega_mean_buf
        with _mp_region("try_patch:omega_gather_and_cost", _dev):
            if candidate_coords.shape == current_nnf.shape:
                candidate_omega_scores, current_omega_scores = gather_omega_scores_pair(
                    omega_mean_map, candidate_coords, current_nnf
                )
            else:
                candidate_omega_scores = gather_omega_scores_from_patches(
                    omega_mean_map, candidate_coords
                )
                current_omega_scores = gather_omega_scores_from_patches(
                    omega_mean_map, current_nnf
                )
            patch_pixel_count = patch_size * patch_size
            inv_norm = 1.0 / (patch_pixel_count * omega_best)
            uf = uniformity_weight * inv_norm
            candidate_total_error = candidate_patch_errors + uf * candidate_omega_scores
            current_total_error = current_errors + uf * current_omega_scores
            update_mask = (
                candidate_total_error < current_total_error
            ) & valid_candidates
    with _mp_region("try_patch:select_write", _dev):
        if update_nnf:
            updated_nnf = torch.where(
                update_mask.unsqueeze(-1),
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
    target_modulation_patches: Optional[torch.Tensor] = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
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

    # OPTIMIZATION: Snapshot NNF before updates (reused buffer per shape/device/dtype).
    with _mp_region("prop:nnf_snapshot", device):
        nnf_original = _prop_nnf_snapshot(nnf)

    omega_mean_map = None
    if uniformity_weight != 0.0:
        with _mp_region("prop:omega_mean_map", device):
            omega_mean_buf = omega_mean_map_buffer(device, H_s, W_s)
            omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
            omega_mean_map = omega_mean_buf

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
    nb = _prop_neighbor_nnf_buf(nnf)
    if shift_x == 1:
        torch.cat((nnf[:, :1, :], nnf[:, :-1, :]), dim=1, out=nb)
    else:
        torch.cat((nnf[:, 1:, :], nnf[:, -1:, :]), dim=1, out=nb)

    # Apply propagation offset
    horiz_candidates = nb + _get_offset_tensor((prop_x, 0), device)

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
        target_modulation_patches,
        use_bilateral,
        sigma_spatial,
        sigma_color,
        n_size_step,
        omega_mean_map=omega_mean_map,
    )

    # OPTIMIZATION: Collect changes but don't update omega yet
    horiz_changed_indices = (horiz_updates & active_mask).nonzero(as_tuple=False)

    # Update NNF and errors for vertical propagation
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)

    # --- Vertical Propagation ---
    if shift_y == 1:
        torch.cat((nnf[:1, :, :], nnf[:-1, :, :]), dim=0, out=nb)
    else:
        torch.cat((nnf[1:, :, :], nnf[-1:, :, :]), dim=0, out=nb)

    vert_candidates = nb + _get_offset_tensor((0, prop_y), device)

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
        target_modulation_patches,
        use_bilateral,
        sigma_spatial,
        sigma_color,
        n_size_step,
        omega_mean_map=omega_mean_map,
    )

    # OPTIMIZATION: Collect vertical changes
    vert_changed_indices = (vert_updates & active_mask).nonzero(as_tuple=False)

    # Update NNF and errors
    nnf.copy_(new_nnf)
    error_map.copy_(new_errors)

    # Omega: horizontal then vertical NNF changes — scatter deltas sum commutatively, but
    # vertical ``old`` coords must be read after ``nnf_original`` is patched for horiz winners.
    with _mp_region("prop:omega_updates", device):
        old_parts = []
        new_parts = []
        if len(horiz_changed_indices) > 0:
            y_h, x_h = horiz_changed_indices[:, 0], horiz_changed_indices[:, 1]
            old_parts.append(nnf_original[y_h, x_h])
            new_parts.append(nnf[y_h, x_h])
            nnf_original[y_h, x_h] = nnf[y_h, x_h]
        if len(vert_changed_indices) > 0:
            y_v, x_v = vert_changed_indices[:, 0], vert_changed_indices[:, 1]
            old_parts.append(nnf_original[y_v, x_v])
            new_parts.append(nnf[y_v, x_v])
        if old_parts:
            if len(old_parts) == 1:
                update_omega_map(
                    omega_map,
                    old_parts[0],
                    new_parts[0],
                    patch_size,
                    clear_device_cache=False,
                )
            else:
                update_omega_map(
                    omega_map,
                    torch.cat(old_parts, dim=0),
                    torch.cat(new_parts, dim=0),
                    patch_size,
                    clear_device_cache=False,
                )


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
    target_modulation_patches: Optional[torch.Tensor] = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
):
    """
    Optimized random search with pruning.
    OPTIMIZED: Reduced omega computations and pre-computed constants.
    """
    device = nnf.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2

    with _mp_region("rs:prep", device):
        # --- 1. Pruning: Select only active pixels ---
        active_mask = mask == 255
        if search_pruning_threshold > 0:
            active_mask &= error_map >= search_pruning_threshold

        active_indices = active_mask.nonzero(as_tuple=True)
        if active_indices[0].numel() == 0:
            return  # Exit early if no pixels to process

        y_coords, x_coords = active_indices
        num_active = y_coords.numel()

        rand_buf_x, rand_buf_y, candidate_buf = _random_search_buffers(num_active, nnf)

        # OPTIMIZATION: Pre-compute constants outside the loop
        patch_pixel_count = patch_size * patch_size
        omega_normalization = patch_pixel_count * omega_best
        uniformity_factor = uniformity_weight / omega_normalization

        # OPTIMIZATION: Pre-extract current NNF for active pixels once
        current_nnf_active = nnf[y_coords, x_coords]
        current_errors_active = error_map[y_coords, x_coords]

        # Target-side data is invariant across random-search radii for this active set.
        target_style_p_v = target_style_patches[y_coords, x_coords]
        target_guide_p_v = (
            target_guide_patches[y_coords, x_coords]
            if target_guide_patches.numel() > 0
            else target_guide_patches
        )
        target_mod_v = (
            target_modulation_patches[y_coords, x_coords]
            if target_modulation_patches is not None
            else None
        )
        target_stats_v = None
        if target_stats is not None:
            t_vals, t_mean, t_std = target_stats
            target_stats_v = (
                t_vals[y_coords, x_coords],
                t_mean[y_coords, x_coords],
                t_std[y_coords, x_coords],
            )
        ssd_kw = dict(
            use_bilateral=use_bilateral,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            n_size_step=n_size_step,
        )
        can_use_fused_ssd = (
            source_guide_patches.numel() > 0
            and (not use_bilateral)
            and n_size_step == 1
        )

        use_omega_cost = uniformity_weight != 0.0
        omega_mean_buf = None
        omega_mean_map = None
        curr_omega_active = None
        if use_omega_cost:
            # Reuse a single mean-map buffer across radius levels (avoids new tensor each rebuild).
            omega_mean_buf = omega_mean_map_buffer(device, H_s, W_s)
            omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
            omega_mean_map = omega_mean_buf
            curr_omega_active = gather_omega_scores_from_patches(
                omega_mean_map, current_nnf_active
            )

    radius = initial_radius
    while radius >= 1:
        with _mp_region("rs:radius_iter", device):
            # --- 2. Generate candidates only for active pixels ---
            if generator is not None:
                torch.randint(
                    -radius,
                    radius + 1,
                    (num_active,),
                    device=device,
                    generator=generator,
                    out=rand_buf_x,
                )
                torch.randint(
                    -radius,
                    radius + 1,
                    (num_active,),
                    device=device,
                    generator=generator,
                    out=rand_buf_y,
                )
            else:
                torch.randint(
                    -radius, radius + 1, (num_active,), device=device, out=rand_buf_x
                )
                torch.randint(
                    -radius, radius + 1, (num_active,), device=device, out=rand_buf_y
                )

            candidate_buf[:, 0].copy_(current_nnf_active[:, 0])
            candidate_buf[:, 1].copy_(current_nnf_active[:, 1])
            candidate_buf[:, 0].add_(rand_buf_x)
            candidate_buf[:, 1].add_(rand_buf_y)
            candidate_coords_active = candidate_buf

            # --- 3. Validate active candidates ---
            valid_mask_active = (
                (candidate_coords_active[..., 0] >= r)
                & (candidate_coords_active[..., 0] < W_s - r)
                & (candidate_coords_active[..., 1] >= r)
                & (candidate_coords_active[..., 1] < H_s - r)
            )

            # --- 4. Compute errors for active candidates ---
            # Keeping the full active set avoids a costly MPS nonzero/compaction
            # per radius; invalid candidates are masked out before any write.
            cand_v = candidate_coords_active

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
                    source_stats,
                    target_stats_v,
                    target_mod_v,
                    use_bilateral=use_bilateral,
                    sigma_spatial=sigma_spatial,
                    sigma_color=sigma_color,
                    n_size_step=n_size_step,
                )
            else:  # SSD
                if can_use_fused_ssd:
                    cand_errors_v = compute_patch_ssd_style_guide_fused(
                        source_style_patches,
                        target_style_p_v,
                        source_guide_patches,
                        target_guide_p_v,
                        cand_v,
                        style_weights,
                        guide_weights,
                        target_mod_v,
                        **ssd_kw,
                    )
                else:
                    cand_errors_v = compute_patch_ssd_vectorized(
                        source_style_patches,
                        target_style_p_v,
                        cand_v,
                        style_weights,
                        None,
                        **ssd_kw,
                    )
                    if source_guide_patches.numel() > 0:
                        cand_errors_v = cand_errors_v + compute_patch_ssd_vectorized(
                            source_guide_patches,
                            target_guide_p_v,
                            cand_v,
                            guide_weights,
                            target_mod_v,
                            source_style_patches_for_bilateral=source_style_patches,
                            **ssd_kw,
                        )

            # --- 5. Compare total error and update ---
            if use_omega_cost:
                cand_omega_v = gather_omega_scores_from_patches(omega_mean_map, cand_v)
                cand_total_error = cand_errors_v + uniformity_factor * cand_omega_v
                curr_total_error = (
                    current_errors_active + uniformity_factor * curr_omega_active
                )
            else:
                cand_total_error = cand_errors_v
                curr_total_error = current_errors_active

            update_mask_v = (cand_total_error < curr_total_error) & valid_mask_active

            update_indices_in_v = update_mask_v.nonzero(as_tuple=True)[0]
            if update_indices_in_v.numel() > 0:
                # Get original coordinates of pixels to update
                final_y = y_coords[update_indices_in_v]
                final_x = x_coords[update_indices_in_v]

                old_nnf_vals = nnf[final_y, final_x]
                new_nnf_vals = cand_v[update_indices_in_v]

                update_omega_map(
                    omega_map,
                    old_nnf_vals,
                    new_nnf_vals,
                    patch_size,
                    clear_device_cache=False,
                )
                nnf[final_y, final_x] = new_nnf_vals.to(nnf.dtype)
                error_map[final_y, final_x] = cand_errors_v[update_indices_in_v]

                # OPTIMIZATION: Update cached current NNF and omega for updated pixels
                current_nnf_active[update_indices_in_v] = new_nnf_vals.to(
                    current_nnf_active.dtype
                )
                if use_omega_cost:
                    curr_omega_active[update_indices_in_v] = cand_omega_v[
                        update_indices_in_v
                    ]
                    omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
                    clear_torch_device_cache(device)
                current_errors_active[update_indices_in_v] = cand_errors_v[
                    update_indices_in_v
                ]

        radius //= 2
