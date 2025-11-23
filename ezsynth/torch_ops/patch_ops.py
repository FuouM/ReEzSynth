# ezsynth/torch_ops/patch_ops.py
"""
Patch extraction and distance computation operations for EBSynth.

This module contains vectorized PyTorch implementations of patch-based
operations that form the core of the PatchMatch algorithm.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

TORCH_CUDA_CLEAR_CACHE = False
TORCH_MPS_CLEAR_CACHE = True


def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extract all patches from an image using unfold.

    Args:
        image: (H, W, C) tensor
        patch_size: Size of square patches (typically 7)

    Returns:
        patches: (H, W, patch_size*patch_size*C) tensor where each position
                contains the flattened patch centered at that pixel
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    H, W, C = image.shape
    padding = patch_size // 2

    image_nchw = image.permute(2, 0, 1).unsqueeze(0)

    is_uint8 = image_nchw.dtype == torch.uint8
    if is_uint8:
        image_nchw = image_nchw.float()

    # Use replicate padding to match CUDA implementation
    image_padded = F.pad(
        image_nchw, (padding, padding, padding, padding), mode="replicate"
    )
    patches = F.unfold(image_padded, kernel_size=patch_size, padding=0)

    if is_uint8:
        patches = patches.round().clamp(0, 255).to(torch.uint8)

    patches = patches.view(C * patch_size * patch_size, H * W)
    patches = patches.permute(1, 0).view(H, W, -1)

    result = patches.contiguous()

    if (
        TORCH_CUDA_CLEAR_CACHE
        and str(image.device).startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()

    return result


def extract_patches_from_coords(
    image: torch.Tensor, coords: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """
    Extract patches from specific coordinates using advanced indexing.
    More memory efficient than unfolding the whole image when we only need a subset.

    Args:
        image: (H, W, C) tensor
        coords: (N, 2) tensor of (x, y) top-left coordinates
        patch_size: Size of patches

    Returns:
        patches: (N, C*ps*ps) flattened patches
    """
    H, W, C = image.shape
    N = coords.shape[0]

    # Ensure coords are long for indexing
    coords = coords.long()

    # Create grid of offsets
    # (ps, ps)
    ys = torch.arange(patch_size, device=image.device)
    xs = torch.arange(patch_size, device=image.device)

    # (ps*ps)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    # Compute all sampling coordinates
    # coords: (N, 2) -> (N, 1)
    # grid: (ps*ps) -> (1, ps*ps)
    # sample_y: (N, ps*ps)
    sample_y = coords[:, 1:2] + grid_y.unsqueeze(0)
    sample_x = coords[:, 0:1] + grid_x.unsqueeze(0)

    # Clamp to be safe (though input coords should be valid for top-left)
    # We need to clamp the *bottom-right* of the patch too
    # But assuming valid top-left coords that allow for a full patch:
    # sample_y will range from y to y+ps-1.
    # If y < H-ps+1, then y+ps-1 < H.
    sample_y = sample_y.clamp(0, H - 1)
    sample_x = sample_x.clamp(0, W - 1)

    # Gather pixels
    # image: (H, W, C)
    # We want (N, ps*ps, C)

    # Advanced indexing with broadcasting
    # sample_y, sample_x are (N, ps*ps)
    # We need to expand them to (N, ps*ps, C)? No, we can index directly if we handle C carefully.
    # Actually, image[sample_y, sample_x] will give (N, ps*ps, C)

    patches = image[sample_y, sample_x]  # (N, ps*ps, C)

    # Flatten to (N, C*ps*ps) to match expected format for some ops,
    # OR keep as (N, C, ps, ps) depending on usage.
    # The existing extract_patches returns (H, W, patch_size*patch_size*C) -> flattened patches.
    # Let's match that "flattened patch" structure: (N, C*ps*ps)
    # But wait, existing extract_patches returns (H, W, -1) where -1 is C*ps*ps.
    # Here we have N patches.

    # patches is (N, ps*ps, C).
    # We need to permute to (N, C, ps*ps) then flatten?
    # Let's check how unfold does it.
    # Unfold (1, C, H, W) -> (1, C*ps*ps, L).
    # So the channel dimension comes first in the flattened vector.
    # i.e. [R0, R1... G0, G1... B0, B1...]

    patches = patches.permute(0, 2, 1)  # (N, C, ps*ps)
    patches = patches.reshape(N, -1)  # (N, C*ps*ps)

    return patches


def compute_patch_ssd_vectorized(
    source_patches: torch.Tensor,
    target_patches: torch.Tensor,
    nnf: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Computes SSD. Handles both 2D grid and 1D list of target patches."""
    H_s, W_s = source_patches.shape[:2]

    # Store original shape and flatten if necessary
    original_shape = target_patches.shape[:-1]
    if target_patches.dim() > 2:
        target_patches = target_patches.flatten(0, -2)
        nnf = nnf.flatten(0, -2)

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)
    matched_source_patches = source_patches[source_y, source_x]

    diff = (matched_source_patches.float() - target_patches.float()) ** 2
    C = len(weights)
    ps_sq = source_patches.shape[2] // C

    diff_reshaped = diff.view(-1, C, ps_sq)
    weighted_diff = diff_reshaped * weights.view(1, -1, 1)
    error = weighted_diff.sum(dim=[1, 2])

    # Reshape back to original if it was a grid
    if len(original_shape) > 1:
        return error.view(original_shape)
    return error


def compute_patch_stats(
    patches: torch.Tensor, patch_size: int, num_channels: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute statistics for NCC: vals (grayscale), mean, std.

    Args:
        patches: (..., C*ps*ps) flattened patches
        patch_size: int
        num_channels: int

    Returns:
        vals: (..., ps*ps) - channel-averaged pixel values
        mean: (..., 1) - mean of vals
        std: (..., 1) - std of vals
    """
    ps_sq = patch_size * patch_size
    epsilon = 1e-6

    # Reshape to (..., C, ps*ps)
    patches_reshaped = patches.view(*patches.shape[:-1], num_channels, ps_sq)

    # Average over channels to get "grayscale" equivalent for NCC
    vals = patches_reshaped.float().mean(dim=-2)  # (..., ps_sq)

    mean = vals.mean(dim=-1, keepdim=True)  # (..., 1)
    std = vals.std(dim=-1, keepdim=True, unbiased=False) + epsilon  # (..., 1)

    return vals, mean, std


def compute_patch_ncc_vectorized(
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    nnf: torch.Tensor,
    patch_size: int,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    source_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    target_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Computes NCC/SSD. Handles both 2D grid and 1D list of target patches."""
    H_s, W_s = source_style_patches.shape[:2]
    C_style = style_weights.shape[0]
    ps_sq = patch_size * patch_size
    epsilon = 1e-6

    # Store original shape and flatten if necessary
    original_shape = target_style_patches.shape[:-1]
    if target_style_patches.dim() > 2:
        target_style_patches = target_style_patches.flatten(0, -2)
        if target_guide_patches.numel() > 0:
            target_guide_patches = target_guide_patches.flatten(0, -2)
        nnf = nnf.flatten(0, -2)

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)
    matched_source_patches = source_style_patches[source_y, source_x]

    # --- NCC Computation (on flattened data) ---
    num_active = matched_source_patches.shape[0]
    matched_source = matched_source_patches.view(num_active, C_style, ps_sq)
    target = target_style_patches.view(num_active, C_style, ps_sq)

    s_vals = matched_source.float().mean(dim=1)
    t_vals = target.float().mean(dim=1)

    mean_s = s_vals.mean(dim=1, keepdim=True)
    mean_t = t_vals.mean(dim=1, keepdim=True)

    std_s = s_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon
    std_t = t_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon

    cov = ((s_vals - mean_s) * (t_vals - mean_t)).mean(dim=1)
    ncc = cov / (std_s.squeeze(1) * std_t.squeeze(1))
    style_error = (1.0 - ncc) * style_weights[0] * float(ps_sq)

    # --- Guide SSD ---
    if source_guide_patches.numel() > 0:
        guide_error = compute_patch_ssd_vectorized(
            source_guide_patches, target_guide_patches, nnf, guide_weights
        )
        total_error = style_error + guide_error
    else:
        total_error = style_error

    # Reshape back to original if it was a grid
    if len(original_shape) > 1:
        return total_error.view(original_shape)
    return total_error
