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


def generate_horizontal_candidates_metal(current_nnf, H_t, W_t, H_s, W_s, is_odd):
    """Generate horizontal neighbor candidates using Metal acceleration."""
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal operations not available")
    return compiled_metal_ops.mps_generate_horizontal_candidates(
        current_nnf, H_t, W_t, H_s, W_s, is_odd
    )


def generate_vertical_candidates_metal(current_nnf, H_t, W_t, H_s, W_s, is_odd):
    """Generate vertical neighbor candidates using Metal acceleration."""
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal operations not available")
    return compiled_metal_ops.mps_generate_vertical_candidates(
        current_nnf, H_t, W_t, H_s, W_s, is_odd
    )


def generate_random_search_candidates_metal(
    current_nnf, random_offsets, H_t, W_t, H_s, W_s
):
    """Generate random search candidates using Metal acceleration."""
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal operations not available")
    return compiled_metal_ops.mps_generate_random_search_candidates(
        current_nnf, random_offsets, H_t, W_t, H_s, W_s
    )


def generate_horizontal_candidates_cpu(current_nnf, H_t, W_t, H_s, W_s, is_odd, device):
    """Generate horizontal neighbor candidates using PyTorch (CPU fallback)."""
    # For odd iteration: try nnf[y, x+1] shifted by (1, 0)
    # For even iteration: try nnf[y, x-1] shifted by (-1, 0)

    if is_odd:
        # Odd iteration: try right neighbor (x+1, y)
        # Shift nnf left and pad right with replication
        horiz_shifted = torch.roll(
            current_nnf, -1, dims=1
        )  # Shift left (x+1 becomes x)
        horiz_shifted[:, -1, :] = horiz_shifted[:, -2, :]  # Replicate last column
        horiz_candidates = horiz_shifted + torch.tensor([1, 0], device=device)
    else:
        # Even iteration: try left neighbor (x-1, y)
        # Shift nnf right and pad left with replication
        horiz_shifted = torch.roll(
            current_nnf, 1, dims=1
        )  # Shift right (x-1 becomes x)
        horiz_shifted[:, 0, :] = horiz_shifted[:, 1, :]  # Replicate first column
        horiz_candidates = horiz_shifted + torch.tensor([-1, 0], device=device)

    return horiz_candidates


def generate_vertical_candidates_cpu(current_nnf, H_t, W_t, H_s, W_s, is_odd, device):
    """Generate vertical neighbor candidates using PyTorch (CPU fallback)."""
    # For odd iteration: try nnf[y+1, x] shifted by (0, 1)
    # For even iteration: try nnf[y-1, x] shifted by (0, -1)

    if is_odd:
        # Odd iteration: try bottom neighbor (x, y+1)
        # Shift nnf up and pad bottom with replication
        vert_shifted = torch.roll(current_nnf, -1, dims=0)  # Shift up (y+1 becomes y)
        vert_shifted[-1, :, :] = vert_shifted[-2, :, :]  # Replicate last row
        vert_candidates = vert_shifted + torch.tensor([0, 1], device=device)
    else:
        # Even iteration: try top neighbor (x, y-1)
        # Shift nnf down and pad top with replication
        vert_shifted = torch.roll(current_nnf, 1, dims=0)  # Shift down (y-1 becomes y)
        vert_shifted[0, :, :] = vert_shifted[1, :, :]  # Replicate first row
        vert_candidates = vert_shifted + torch.tensor([0, -1], device=device)

    return vert_candidates


def try_patch_batch(
    candidate_coords: torch.Tensor,  # (H_t, W_t, 2) int32 - candidate source coords
    current_nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    current_errors: torch.Tensor,  # (H_t, W_t) float32
    omega_map: torch.Tensor,  # (H_s, W_s) int32
    source_style: torch.Tensor,  # (H_s, W_s, C_s) uint8
    target_style: torch.Tensor,  # (H_t, W_t, C_s) uint8
    source_guide: torch.Tensor,  # (H_s, W_s, C_g) uint8
    target_guide: torch.Tensor,  # (H_t, W_t, C_g) uint8
    style_weights: torch.Tensor,  # (C_s,) float32
    guide_weights: torch.Tensor,  # (C_g,) float32
    uniformity_weight: float,
    patch_size: int,
    cost_function_mode: int,
    omega_best: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try candidate patches and update NNF if better.

    Args:
        candidate_coords: Candidate source coordinates to try
        current_nnf: Current best NNF mapping
        current_errors: Current error values
        omega_map: Current patch usage counts
        source_style: Source style image
        target_style: Target style image
        source_guide: Source guide channels
        target_guide: Target guide channels
        style_weights: Per-channel style weights
        guide_weights: Per-channel guide weights
        uniformity_weight: Weight for uniformity penalty
        patch_size: Size of patches
        cost_function_mode: COST_FUNCTION_SSD or COST_FUNCTION_NCC
        omega_best: Ideal omega value for normalization

    Returns:
        updated_nnf: (H_t, W_t, 2) updated NNF
        updated_errors: (H_t, W_t) updated error values
        update_mask: (H_t, W_t) bool - which positions were updated

    Algorithm:
    1. Validate candidate coordinates (must be within patch bounds)
    2. Compute patch errors for candidates
    3. Compute omega scores (uniformity penalty)
    4. Compare total error: patch_error + uniformity_weight * omega_score
    5. Update NNF where candidates are better
    """
    H_t, W_t = target_style.shape[:2]
    H_s, W_s = source_style.shape[:2]

    # Validate candidate coordinates - must be far enough from edges for patch
    r = patch_size // 2
    valid_candidates = (
        (candidate_coords[..., 0] >= r)
        & (candidate_coords[..., 0] < W_s - r)
        & (candidate_coords[..., 1] >= r)
        & (candidate_coords[..., 1] < H_s - r)
    )

    # Compute candidate patch errors
    if cost_function_mode == COST_FUNCTION_NCC:
        candidate_patch_errors = compute_patch_ncc_vectorized(
            source_style,
            target_style,
            source_guide,
            target_guide,
            candidate_coords,
            patch_size,
            style_weights,
            guide_weights,
        )
    else:  # SSD
        # For SSD, we need pre-computed patches
        from .patch_ops import extract_patches

        source_patches = extract_patches(source_style, patch_size)
        target_patches = extract_patches(target_style, patch_size)
        candidate_patch_errors = compute_patch_ssd_vectorized(
            source_patches, target_patches, candidate_coords, style_weights
        )

        # Add guide errors if present
        if source_guide.shape[2] > 0:
            source_guide_patches = extract_patches(source_guide, patch_size)
            target_guide_patches = extract_patches(target_guide, patch_size)
            guide_errors = compute_patch_ssd_vectorized(
                source_guide_patches,
                target_guide_patches,
                candidate_coords,
                guide_weights,
            )
            candidate_patch_errors += guide_errors

    # Compute omega scores for candidates and current mappings
    candidate_omega_scores = compute_omega_scores(
        omega_map, candidate_coords, patch_size
    )
    current_omega_scores = compute_omega_scores(omega_map, current_nnf, patch_size)

    # Normalize omega scores
    patch_pixel_count = patch_size * patch_size
    candidate_omega_normalized = candidate_omega_scores / (
        patch_pixel_count * omega_best
    )
    current_omega_normalized = current_omega_scores / (patch_pixel_count * omega_best)

    # Total errors including uniformity penalty
    candidate_total_error = (
        candidate_patch_errors + uniformity_weight * candidate_omega_normalized
    )
    current_total_error = current_errors + uniformity_weight * current_omega_normalized

    # Determine which candidates are better
    update_mask = (candidate_total_error < current_total_error) & valid_candidates

    # Update NNF and errors
    updated_nnf = torch.where(
        update_mask.unsqueeze(2).expand_as(current_nnf), candidate_coords, current_nnf
    )

    updated_errors = torch.where(update_mask, candidate_patch_errors, current_errors)

    return updated_nnf, updated_errors, update_mask


def propagation_step(
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32, updated in-place
    error_map: torch.Tensor,  # (H_t, W_t) float32, updated in-place
    omega_map: torch.Tensor,  # (H_s, W_s) int32, updated in-place
    source_style: torch.Tensor,  # (H_s, W_s, C_s) uint8
    target_style: torch.Tensor,  # (H_t, W_t, C_s) uint8
    source_guide: torch.Tensor,  # (H_s, W_s, C_g) uint8
    target_guide: torch.Tensor,  # (H_t, W_t, C_g) uint8
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    is_odd: bool,
    mask: torch.Tensor,  # (H_t, W_t) uint8
    cost_function_mode: int,
    omega_best: float,
):
    """
    Propagation step: try neighbors' matches for spatial coherence.

    Direction depends on is_odd:
    - is_odd=True: scan top-left to bottom-right, try left/up neighbors
    - is_odd=False: scan bottom-right to top-left, try right/down neighbors

    Args:
        All tensors are modified in-place where updates occur.
        mask: Active region (255 = active, 0 = converged)
        is_odd: Direction flag
        Other params: same as try_patch_batch
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_style.shape[:2]
    device = nnf.device

    # Only process active (masked) pixels
    active_mask = mask == 255

    # For even iterations, reverse processing order by flipping tensors
    if not is_odd:
        nnf = torch.flip(nnf, dims=[0, 1])
        error_map = torch.flip(error_map, dims=[0, 1])
        target_style = torch.flip(target_style, dims=[0, 1])
        target_guide = torch.flip(target_guide, dims=[0, 1])
        active_mask = torch.flip(active_mask, dims=[0, 1])

    # Try horizontal neighbor
    # First compute horiz_shifted (needed for omega map updates)
    if is_odd:
        # Odd iteration: try right neighbor (x+1, y)
        horiz_shifted = torch.roll(nnf, -1, dims=1)  # Shift left (x+1 becomes x)
        horiz_shifted[:, -1, :] = horiz_shifted[:, -2, :]  # Replicate last column
    else:
        # Even iteration: try left neighbor (x-1, y)
        horiz_shifted = torch.roll(nnf, 1, dims=1)  # Shift right (x-1 becomes x)
        horiz_shifted[:, 0, :] = horiz_shifted[:, 1, :]  # Replicate first column

    if (
        torch.backends.mps.is_available()
        and METAL_AVAILABLE
        and nnf.device.type == "mps"
    ):
        # Use Metal acceleration for neighbor generation
        try:
            horiz_candidates = generate_horizontal_candidates_metal(
                nnf, H_t, W_t, H_s, W_s, is_odd
            )
        except RuntimeError as e:
            print(f"Metal horizontal candidates failed: {e}, falling back to CPU")
            # Fall back to CPU implementation if Metal fails
            horiz_candidates = generate_horizontal_candidates_cpu(
                nnf, H_t, W_t, H_s, W_s, is_odd, device
            )
    else:
        # CPU implementation
        horiz_candidates = generate_horizontal_candidates_cpu(
            nnf, H_t, W_t, H_s, W_s, is_odd, device
        )

    # Try horizontal candidates
    nnf, error_map, horiz_updates = try_patch_batch(
        horiz_candidates,
        nnf,
        error_map,
        omega_map,
        source_style,
        target_style,
        source_guide,
        target_guide,
        style_weights,
        guide_weights,
        uniformity_weight,
        patch_size,
        cost_function_mode,
        omega_best,
    )

    # Update omega map for changed positions
    changed_indices = (horiz_updates & active_mask).nonzero(as_tuple=False)
    if len(changed_indices) > 0:
        y_changed, x_changed = changed_indices[:, 0], changed_indices[:, 1]
        old_coords = horiz_shifted[y_changed, x_changed]
        new_coords = nnf[y_changed, x_changed]
        update_omega_map(omega_map, old_coords, new_coords, patch_size)

    # Try vertical neighbor
    # First compute vert_shifted (needed for omega map updates)
    if is_odd:
        # Odd iteration: try bottom neighbor (x, y+1)
        vert_shifted = torch.roll(nnf, -1, dims=0)  # Shift up (y+1 becomes y)
        vert_shifted[-1, :, :] = vert_shifted[-2, :, :]  # Replicate last row
    else:
        # Even iteration: try top neighbor (x, y-1)
        vert_shifted = torch.roll(nnf, 1, dims=0)  # Shift down (y-1 becomes y)
        vert_shifted[0, :, :] = vert_shifted[1, :, :]  # Replicate first row

    if (
        torch.backends.mps.is_available()
        and METAL_AVAILABLE
        and nnf.device.type == "mps"
    ):
        # Use Metal acceleration for neighbor generation
        try:
            vert_candidates = generate_vertical_candidates_metal(
                nnf, H_t, W_t, H_s, W_s, is_odd
            )
        except RuntimeError as e:
            print(f"Metal vertical candidates failed: {e}, falling back to CPU")
            # Fall back to CPU implementation if Metal fails
            vert_candidates = generate_vertical_candidates_cpu(
                nnf, H_t, W_t, H_s, W_s, is_odd, device
            )
    else:
        # CPU implementation
        vert_candidates = generate_vertical_candidates_cpu(
            nnf, H_t, W_t, H_s, W_s, is_odd, device
        )

    # Try vertical candidates
    nnf, error_map, vert_updates = try_patch_batch(
        vert_candidates,
        nnf,
        error_map,
        omega_map,
        source_style,
        target_style,
        source_guide,
        target_guide,
        style_weights,
        guide_weights,
        uniformity_weight,
        patch_size,
        cost_function_mode,
        omega_best,
    )

    # Update omega map for changed positions
    changed_indices = (vert_updates & active_mask).nonzero(as_tuple=False)
    if len(changed_indices) > 0:
        y_changed, x_changed = changed_indices[:, 0], changed_indices[:, 1]
        old_coords = vert_shifted[y_changed, x_changed]
        new_coords = nnf[y_changed, x_changed]
        update_omega_map(omega_map, old_coords, new_coords, patch_size)

    # Flip tensors back if needed
    if not is_odd:
        nnf = torch.flip(nnf, dims=[0, 1])
        error_map = torch.flip(error_map, dims=[0, 1])


def random_search_step(
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32, updated in-place
    error_map: torch.Tensor,  # (H_t, W_t) float32, updated in-place
    omega_map: torch.Tensor,  # (H_s, W_s) int32, updated in-place
    source_style: torch.Tensor,
    target_style: torch.Tensor,
    source_guide: torch.Tensor,
    target_guide: torch.Tensor,
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
    Random search: try random offsets with exponentially decreasing radius.

    Algorithm:
    - radius = initial_radius
    - while radius >= 1:
      - Generate random offsets for all pixels: (-radius, radius)
      - candidates = current_nnf + random_offsets
      - Try candidates and update NNF
      - radius //= 2

    Args:
        initial_radius: Starting search radius (typically max(H_s, W_s) // 2)
        search_pruning_threshold: Skip pixels with error below this threshold
        generator: Random number generator for reproducibility
        Other params: same as propagation_step
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_style.shape[:2]
    device = nnf.device

    # Active mask: only process non-converged pixels
    active_mask = mask == 255

    # Pruning: skip pixels with low error (likely already good matches)
    if search_pruning_threshold > 0:
        active_mask &= error_map >= search_pruning_threshold

    radius = initial_radius

    while radius >= 1:
        # Generate random offsets for all pixels
        random_offsets_x = torch.randint(
            -radius, radius + 1, (H_t, W_t), device=device, generator=generator
        )
        random_offsets_y = torch.randint(
            -radius, radius + 1, (H_t, W_t), device=device, generator=generator
        )
        random_offsets = torch.stack([random_offsets_x, random_offsets_y], dim=-1)

        # Candidate coordinates: current NNF + random offsets
        if (
            torch.backends.mps.is_available()
            and METAL_AVAILABLE
            and nnf.device.type == "mps"
            and random_offsets.device.type == "mps"
        ):
            # Use Metal acceleration for candidate generation with boundary validation
            try:
                candidates = generate_random_search_candidates_metal(
                    nnf, random_offsets, H_t, W_t, H_s, W_s
                )
            except RuntimeError as e:
                print(
                    f"Metal random search candidates failed: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation if Metal fails
                candidates = nnf + random_offsets
        else:
            # CPU implementation
            candidates = nnf + random_offsets

        # Try candidates
        new_nnf, new_errors, updates = try_patch_batch(
            candidates,
            nnf,
            error_map,
            omega_map,
            source_style,
            target_style,
            source_guide,
            target_guide,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            cost_function_mode,
            omega_best,
        )

        # Update omega map for changed positions
        changed_indices = (updates & active_mask).nonzero(as_tuple=False)
        if len(changed_indices) > 0:
            y_changed, x_changed = changed_indices[:, 0], changed_indices[:, 1]
            old_coords = nnf[y_changed, x_changed]
            new_coords = new_nnf[y_changed, x_changed]
            update_omega_map(omega_map, old_coords, new_coords, patch_size)

        # Update in-place
        nnf.copy_(new_nnf)
        error_map.copy_(new_errors)

        # Halve radius for next iteration
        radius //= 2
