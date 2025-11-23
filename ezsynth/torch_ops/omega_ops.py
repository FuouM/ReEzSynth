# ezsynth/torch_ops/omega_ops.py
"""
Omega map operations for tracking patch usage in EBSynth.

The omega map tracks how many times each source pixel is used in the current
NNF, enabling a uniformity penalty that discourages overuse of popular patches.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def populate_omega_map(
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    source_shape: Tuple[int, int],  # (H_s, W_s)
    patch_size: int,
) -> torch.Tensor:
    """
    Count how many target patches reference each source pixel.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation.

    For each NNF entry pointing to a source patch center, increment counts
    for all pixels covered by that patch (patch_size x patch_size region).

    Args:
        nnf: Nearest neighbor field mapping target pixels to source coordinates
        source_shape: (H_s, W_s) dimensions of source image
        patch_size: Size of patches (odd number)

    Returns:
        omega_map: (H_s, W_s) int32 tensor with usage counts

    Implementation Strategy:
    - For each NNF entry, expand to cover patch_size x patch_size region
    - Use scatter_add to efficiently accumulate counts
    - Clamp coordinates to valid range
    """

    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_shape
    device = nnf.device

    omega_map = torch.zeros((H_s, W_s), dtype=torch.int32, device=device)

    r = patch_size // 2

    # Create offset grid for patch coverage: all relative positions within a patch
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x, offsets_y], dim=-1)  # (ps, ps, 2)

    # Expand NNF to all patch pixels
    # nnf: (H_t, W_t, 2) -> (H_t, W_t, 1, 1, 2) + (1, 1, ps, ps, 2) = (H_t, W_t, ps, ps, 2)
    expanded_coords = nnf.unsqueeze(2).unsqueeze(3) + offsets.unsqueeze(0).unsqueeze(0)

    # Flatten to list of all coordinates: (H_t * W_t * ps * ps, 2)
    all_coords = expanded_coords.reshape(-1, 2)

    # Clamp to valid range
    all_coords[:, 0].clamp_(0, W_s - 1)
    all_coords[:, 1].clamp_(0, H_s - 1)

    # Convert to linear indices for scatter_add
    linear_indices = all_coords[:, 1] * W_s + all_coords[:, 0]

    # Count occurrences using bincount, then reshape to (H_s, W_s)
    counts = torch.bincount(linear_indices, minlength=H_s * W_s)
    omega_map = counts.view(H_s, W_s).to(torch.int32)

    return omega_map


def compute_omega_scores(
    omega_map: torch.Tensor,  # (H_s, W_s) int32
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
) -> torch.Tensor:
    """
    Compute average omega value for each patch in the NNF.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation.

    Args:
        omega_map: Usage counts for each source pixel
        nnf: Current nearest neighbor field
        patch_size: Size of patches

    Returns:
        omega_scores: (H_t, W_t) float32 average omega values per patch
    """

    H_t, W_t = nnf.shape[:2]
    H_s, W_s = omega_map.shape

    r = patch_size // 2

    # Extract patches from omega map using unfold
    omega_float = omega_map.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H_s, W_s)
    omega_patches = F.unfold(
        omega_float, kernel_size=patch_size, padding=r
    )  # (1, ps*ps, H_s*W_s)

    omega_patches = omega_patches.view(patch_size * patch_size, H_s, W_s)
    omega_patches = omega_patches.permute(1, 2, 0)  # (H_s, W_s, ps*ps)

    # Clamp NNF coordinates to valid range
    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)

    # Gather patches for NNF locations
    matched_omega = omega_patches[source_y, source_x]  # (H_t, W_t, ps*ps)

    # Average over patch pixels
    omega_scores = matched_omega.mean(dim=2)  # (H_t, W_t)

    return omega_scores


def update_omega_map(
    omega_map: torch.Tensor,  # (H_s, W_s) int32, modified in-place
    old_coords: torch.Tensor,  # (N, 2) int32 - coordinates to decrement
    new_coords: torch.Tensor,  # (N, 2) int32 - coordinates to increment
    patch_size: int,
):
    """
    Update omega map when NNF entries change.

    When a target pixel changes from pointing to one source patch to another,
    we need to decrement the old patch region and increment the new one.

    Args:
        omega_map: Current usage counts, modified in-place
        old_coords: Source coordinates to decrement (N, 2)
        new_coords: Source coordinates to increment (N, 2)
        patch_size: Size of patches

    Implementation:
    - Expand coordinates to cover patch regions
    - Use scatter_add to efficiently update counts
    """
    H_s, W_s = omega_map.shape
    device = omega_map.device
    # N = old_coords.shape[0]

    r = patch_size // 2
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x, offsets_y], dim=-1)  # (ps, ps, 2)

    # Expand coordinates to patch coverage
    old_expanded = old_coords.unsqueeze(1).unsqueeze(2) + offsets  # (N, ps, ps, 2)
    new_expanded = new_coords.unsqueeze(1).unsqueeze(2) + offsets

    old_flat = old_expanded.reshape(-1, 2)  # (N*ps*ps, 2)
    new_flat = new_expanded.reshape(-1, 2)

    # Clamp coordinates
    old_flat[:, 0].clamp_(0, W_s - 1)
    old_flat[:, 1].clamp_(0, H_s - 1)
    new_flat[:, 0].clamp_(0, W_s - 1)
    new_flat[:, 1].clamp_(0, H_s - 1)

    # Convert to linear indices
    old_linear = old_flat[:, 1] * W_s + old_flat[:, 0]
    new_linear = new_flat[:, 1] * W_s + new_flat[:, 0]

    # Update counts: decrement old, increment new
    omega_flat = omega_map.flatten()
    omega_flat.scatter_add_(
        0, old_linear, torch.full_like(old_linear, -1, dtype=omega_flat.dtype)
    )
    omega_flat.scatter_add_(
        0, new_linear, torch.full_like(new_linear, 1, dtype=omega_flat.dtype)
    )
    omega_map.copy_(omega_flat.view(H_s, W_s))
