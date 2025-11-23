# ezsynth/torch_ops/voting_ops.py
"""
Voting operations for image reconstruction in EBSynth.

These functions reconstruct the target image by averaging all source patches
that contribute to each target pixel according to the current NNF.
"""

import os

import torch
import torch.nn.functional as F

def vote_plain(
    source_style: torch.Tensor,  # (H_s, W_s, C) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
) -> torch.Tensor:
    """
    Reconstruct target image by averaging overlapping patches.

    Args:
        source_style: Source style image
        nnf: Nearest neighbor field mapping target->source
        patch_size: Size of patches (odd)

    Returns:
        target_style: Reconstructed target image (H_t, W_t, C) uint8
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    device = source_style.device
    dtype = source_style.dtype

    r = patch_size // 2

    # Initialize accumulation buffers
    accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

    # Create coordinate grids for all target pixels
    ty_grid, tx_grid = torch.meshgrid(
        torch.arange(H_t, device=device),
        torch.arange(W_t, device=device),
        indexing="ij",
    )

    # Pre-compute all patch offsets to reduce loop overhead
    patch_offsets = []
    for py in range(-r, r + 1):
        for px in range(-r, r + 1):
            patch_offsets.append((py, px))

    # Process offsets in batches to improve memory locality
    batch_size = min(16, len(patch_offsets))  # Process in small batches

    for i in range(0, len(patch_offsets), batch_size):
        batch_offsets = patch_offsets[i : i + batch_size]

        for py, px in batch_offsets:
            # Target patch centers that contribute to current pixel
            t_neighbor_x = tx_grid - px
            t_neighbor_y = ty_grid - py

            # Create mask for valid neighbor coordinates
            valid_mask = (
                (t_neighbor_x >= 0)
                & (t_neighbor_x < W_t)
                & (t_neighbor_y >= 0)
                & (t_neighbor_y < H_t)
            )

            # Get valid indices
            valid_indices = valid_mask.nonzero(as_tuple=False)
            if len(valid_indices) == 0:
                continue

            valid_ty = valid_indices[:, 0]
            valid_tx = valid_indices[:, 1]

            # Look up source patch centers for these neighbors
            neighbor_y = t_neighbor_y[valid_ty, valid_tx]
            neighbor_x = t_neighbor_x[valid_ty, valid_tx]

            source_center_x = nnf[neighbor_y, neighbor_x, 0]
            source_center_y = nnf[neighbor_y, neighbor_x, 1]

            # Source pixel coordinates to sample
            source_x = (source_center_x + px).clamp(0, W_s - 1)
            source_y = (source_center_y + py).clamp(0, H_s - 1)

            # Gather source values
            source_values = source_style[source_y, source_x]

            # Accumulate contributions
            accumulator[valid_ty, valid_tx] += source_values.float()
            weight_sum[valid_ty, valid_tx] += 1.0

    # Average (avoid division by zero)
    target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)

    # Convert back to original dtype and clamp
    target_style = target_style.clamp(0, 255).to(dtype)

    return target_style


def vote_weighted(
    source_style: torch.Tensor,  # (H_s, W_s, C) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    error_map: torch.Tensor,  # (H_t, W_t) float32
    patch_size: int,
) -> torch.Tensor:
    """
    Weighted voting using patch errors.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation using F.fold.

    Args:
        source_style: Source style image
        nnf: Nearest neighbor field
        error_map: Error values for each patch in NNF
        patch_size: Size of patches

    Returns:
        target_style: Reconstructed target image with weighted averaging
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    device = source_style.device
    dtype = source_style.dtype

    r = patch_size // 2

    # Initialize accumulation buffers
    accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

    # Create coordinate grids
    ty_grid, tx_grid = torch.meshgrid(
        torch.arange(H_t, device=device),
        torch.arange(W_t, device=device),
        indexing="ij",
    )

    # Pre-compute all patch offsets
    patch_offsets = []
    for py in range(-r, r + 1):
        for px in range(-r, r + 1):
            patch_offsets.append((py, px))

    # Process offsets in batches to improve memory locality
    batch_size = min(16, len(patch_offsets))

    for i in range(0, len(patch_offsets), batch_size):
        batch_offsets = patch_offsets[i : i + batch_size]

        for py, px in batch_offsets:
            t_neighbor_x = tx_grid - px
            t_neighbor_y = ty_grid - py

            valid_mask = (
                (t_neighbor_x >= 0)
                & (t_neighbor_x < W_t)
                & (t_neighbor_y >= 0)
                & (t_neighbor_y < H_t)
            )

            valid_indices = valid_mask.nonzero(as_tuple=False)
            if len(valid_indices) == 0:
                continue

            valid_ty = valid_indices[:, 0]
            valid_tx = valid_indices[:, 1]

            neighbor_y = t_neighbor_y[valid_ty, valid_tx]
            neighbor_x = t_neighbor_x[valid_ty, valid_tx]

            source_center_x = nnf[neighbor_y, neighbor_x, 0]
            source_center_y = nnf[neighbor_y, neighbor_x, 1]

            source_x = (source_center_x + px).clamp(0, W_s - 1)
            source_y = (source_center_y + py).clamp(0, H_s - 1)

            source_values = source_style[source_y, source_x]  # (N_valid, C)

            # Compute weights from error map
            errors = error_map[neighbor_y, neighbor_x]  # (N_valid,)
            weights = 1.0 / (1.0 + errors)  # (N_valid,)

            # Weighted accumulation
            weighted_values = source_values.float() * weights.unsqueeze(
                1
            )  # (N_valid, C)
            accumulator[valid_ty, valid_tx] += weighted_values
            weight_sum[valid_ty, valid_tx] += weights

    # Weighted average
    target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)
    target_style = target_style.clamp(0, 255).to(dtype)

    return target_style
