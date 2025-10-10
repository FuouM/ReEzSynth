# ezsynth/torch_ops/patch_ops.py
"""
Patch extraction and distance computation operations for EBSynth.

This module contains vectorized PyTorch implementations of patch-based
operations that form the core of the PatchMatch algorithm.
"""

import os

import torch
import torch.nn.functional as F

# Check if Metal operations should be skipped
skip_metal = os.environ.get("EZSYNTH_SKIP_METAL", "").lower() in ("1", "true", "yes")
skip_metal_verbose = os.environ.get("EZSYNTH_SKIP_METAL_VERBOSE", "").lower() in (
    "1",
    "true",
    "yes",
)
# Import custom Metal operations if MPS is available and not skipped
if torch.backends.mps.is_available() and not skip_metal:
    try:
        from .metal_ext.compiler import compiled_metal_ops

        if compiled_metal_ops is not None:
            if skip_metal_verbose:
                print("Loaded custom Metal operations for patch_ops.")
        else:
            if skip_metal_verbose:
                print("Custom Metal operations compiler returned None for patch_ops.")
            compiled_metal_ops = None
    except ImportError:
        if skip_metal_verbose:
            print("Failed to import custom Metal operations for patch_ops.")
        compiled_metal_ops = None
else:
    if skip_metal:
        if skip_metal_verbose:
            print(
                "Skipping custom Metal operations for patch_ops (EZSYNTH_SKIP_METAL set)."
            )
    compiled_metal_ops = None


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

    patches = F.unfold(image_nchw, kernel_size=patch_size, padding=padding)

    if is_uint8:
        patches = patches.round().clamp(0, 255).to(torch.uint8)

    patches = patches.view(C * patch_size * patch_size, H * W)
    patches = patches.permute(1, 0).view(H, W, -1)

    return patches.contiguous()


def compute_patch_ssd_vectorized(
    source_patches: torch.Tensor,  # (H_s, W_s, C*ps*ps)
    target_patches: torch.Tensor,  # (H_t, W_t, C*ps*ps)
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    weights: torch.Tensor,  # (C,) float32
) -> torch.Tensor:
    """
    Compute SSD between target patches and their NNF-matched source patches
    using PRE-COMPUTED patches.
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_patches.shape[:2]

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)

    matched_source_patches = source_patches[source_y, source_x]

    diff = (matched_source_patches.float() - target_patches.float()) ** 2

    C = len(weights)
    ps_sq = source_patches.shape[2] // C

    diff_reshaped = diff.view(H_t, W_t, C, ps_sq)
    weighted_diff = diff_reshaped * weights.view(1, 1, -1, 1)
    error = weighted_diff.sum(dim=[2, 3])

    return error


def compute_patch_ncc_vectorized(
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    nnf: torch.Tensor,
    patch_size: int,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute NCC for style and SSD for guides using PRE-COMPUTED patches.
    """
    H_t, W_t = target_style_patches.shape[:2]
    H_s, W_s = source_style_patches.shape[:2]
    C_style = style_weights.shape[0]
    ps_sq = patch_size * patch_size
    epsilon = 1e-6

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)
    matched_source_patches = source_style_patches[source_y, source_x]

    matched_source = matched_source_patches.view(H_t, W_t, C_style, ps_sq)
    target = target_style_patches.view(H_t, W_t, C_style, ps_sq)

    s_vals = matched_source.float().mean(dim=2)
    t_vals = target.float().mean(dim=2)

    mean_s = s_vals.mean(dim=2, keepdim=True)
    mean_t = t_vals.mean(dim=2, keepdim=True)

    std_s = s_vals.std(dim=2, keepdim=True, unbiased=False) + epsilon
    std_t = t_vals.std(dim=2, keepdim=True, unbiased=False) + epsilon

    cov = ((s_vals - mean_s) * (t_vals - mean_t)).mean(dim=2)

    ncc = cov / (std_s.squeeze(2) * std_t.squeeze(2))
    style_error = (1.0 - ncc) * style_weights[0] * float(ps_sq)

    if source_guide_patches.numel() > 0:
        guide_error = compute_patch_ssd_vectorized(
            source_guide_patches, target_guide_patches, nnf, guide_weights
        )
        return style_error + guide_error
    else:
        return style_error
