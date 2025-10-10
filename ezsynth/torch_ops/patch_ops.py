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

# Note: torch.compile disabled due to MPS compatibility issues with complex indexing
# TODO: Re-enable torch.compile for CUDA when Triton kernels are available


def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extract all patches from an image using unfold.

    Args:
        image: (H, W, C) tensor
        patch_size: Size of square patches (typically 7)

    Returns:
        patches: (H, W, patch_size*patch_size*C) tensor where each position
                contains the flattened patch centered at that pixel

    Implementation Notes:
    - unfold expects (N, C, H, W) format, so we permute
    - unfold doesn't support uint8, so we convert to float and back
    - Returns (N, C*ps*ps, H*W) which we reshape to (H, W, C*ps*ps)
    - Padding ensures patches at boundaries are valid
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    H, W, C = image.shape
    padding = patch_size // 2

    # Convert to NCHW format for unfold
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # unfold doesn't support uint8, convert to float
    is_uint8 = image_nchw.dtype == torch.uint8
    if is_uint8:
        image_nchw = image_nchw.float()

    # Extract patches: (1, C*ps*ps, H*W)
    patches = F.unfold(image_nchw, kernel_size=patch_size, padding=padding)

    # Convert back to uint8 if needed
    if is_uint8:
        patches = patches.round().clamp(0, 255).to(torch.uint8)

    # Reshape to (H, W, C*ps*ps)
    # patches is (1, C*ps*ps, H*W) -> (C*ps*ps, H*W) -> (H, W, C*ps*ps)
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
    Compute SSD between target patches and their NNF-matched source patches.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation.

    Args:
        source_patches: All source patches, shape (H_s, W_s, C*ps*ps)
        target_patches: All target patches, shape (H_t, W_t, C*ps*ps)
        nnf: Nearest neighbor field, shape (H_t, W_t, 2) with (x, y) coords
        weights: Per-channel weights, shape (C,)

    Returns:
        error: (H_t, W_t) float32 error map

    Implementation Strategy:
    - Use Metal kernels for acceleration on MPS, fallback to PyTorch
    - Compute weighted squared differences between matched patches
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_patches.shape[:2]
    C = len(weights)

    if torch.backends.mps.is_available() and compiled_metal_ops is not None:
        # Flatten the patches for the Metal kernel
        source_patches_flat = (
            source_patches[nnf[..., 1], nnf[..., 0]]
            .reshape(-1, C * (source_patches.shape[2] // C))
            .float()
            .contiguous()
        )
        target_patches_flat = (
            target_patches.reshape(-1, C * (target_patches.shape[2] // C))
            .float()
            .contiguous()
        )

        # Ensure weights are contiguous
        weights_flat = weights.contiguous()

        # Call the Metal accelerated function (now uses async execution with MPS synchronization)
        error = compiled_metal_ops.mps_compute_patch_ssd(
            source_patches_flat, target_patches_flat, weights_flat
        ).reshape(H_t, W_t)
        return error
    else:
        # Extract source patch locations from NNF
        source_x = nnf[..., 0].clamp(0, W_s - 1)  # (H_t, W_t)
        source_y = nnf[..., 1].clamp(0, H_s - 1)  # (H_t, W_t)

        # Gather source patches using advanced indexing
        matched_source_patches = source_patches[
            source_y, source_x
        ]  # (H_t, W_t, C*ps*ps)

        # Compute weighted SSD
        diff = (
            matched_source_patches.float() - target_patches.float()
        ) ** 2  # (H_t, W_t, C*ps*ps)

        # Apply per-channel weights (need to expand weights to match patch dimensions)
        C = len(weights)
        ps_sq = source_patches.shape[2] // C

        # Reshape to (H_t, W_t, C, ps*ps)
        diff_reshaped = diff.view(H_t, W_t, C, ps_sq)

        # Apply weights: (H_t, W_t, C, ps*ps) * (C, 1) -> (H_t, W_t, C, ps*ps)
        weighted_diff = diff_reshaped * weights.view(1, 1, -1, 1)

        # Sum over all dimensions except spatial
        error = weighted_diff.sum(dim=[2, 3])  # (H_t, W_t)

        return error


def compute_patch_ncc_vectorized_tiled(
    source_style: torch.Tensor,  # (H_s, W_s, C_style) uint8
    target_style: torch.Tensor,  # (H_t, W_t, C_style) uint8
    source_guide: torch.Tensor,  # (H_s, W_s, C_guide) uint8
    target_guide: torch.Tensor,  # (H_t, W_t, C_guide) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
    style_weights: torch.Tensor,  # (C_style,) float32
    guide_weights: torch.Tensor,  # (C_guide,) float32
    tile_size: int = 256,
) -> torch.Tensor:
    """
    Tiled version of NCC computation to reduce memory usage.

    Processes the target image in tiles to avoid creating large patch tensors.
    """
    H_t, W_t = target_style.shape[:2]
    device = target_style.device

    # Initialize result tensor
    result = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

    # Process in tiles
    for y_start in range(0, H_t, tile_size):
        for x_start in range(0, W_t, tile_size):
            y_end = min(y_start + tile_size, H_t)
            x_end = min(x_start + tile_size, W_t)

            # Extract tile from target
            target_style_tile = target_style[y_start:y_end, x_start:x_end]
            target_guide_tile = target_guide[y_start:y_end, x_start:x_end]
            nnf_tile = nnf[y_start:y_end, x_start:x_end]

            # Compute NCC for this tile
            tile_result = compute_patch_ncc_vectorized(
                source_style,
                target_style_tile,
                source_guide,
                target_guide_tile,
                nnf_tile,
                patch_size,
                style_weights,
                guide_weights,
            )

            # Store result
            result[y_start:y_end, x_start:x_end] = tile_result

    return result


def compute_patch_ncc_vectorized(
    source_style: torch.Tensor,  # (H_s, W_s, C_style) uint8
    target_style: torch.Tensor,  # (H_t, W_t, C_style) uint8
    source_guide: torch.Tensor,  # (H_s, W_s, C_guide) uint8
    target_guide: torch.Tensor,  # (H_t, W_t, C_guide) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
    style_weights: torch.Tensor,  # (C_style,) float32
    guide_weights: torch.Tensor,  # (C_guide,) float32
) -> torch.Tensor:
    """
    Compute NCC for style and SSD for guides.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation.

    NCC Formula: ncc = cov(X, Y) / (std(X) * std(Y))
    Cost = (1 - ncc) * scale_factor

    Args:
        source_style: Source style image
        target_style: Target style image
        source_guide: Source guide channels (edges, depth, etc.)
        target_guide: Target guide channels
        nnf: Nearest neighbor field
        patch_size: Size of patches
        style_weights: Weights for style channels
        guide_weights: Weights for guide channels

    Returns:
        error: (H_t, W_t) float32 combined error map
    """
    H_t, W_t = target_style.shape[:2]
    epsilon = 1e-6

    # Extract patches for both style and guide
    source_style_patches = extract_patches(
        source_style, patch_size
    )  # (H_s, W_s, C_s*ps*ps)
    target_style_patches = extract_patches(
        target_style, patch_size
    )  # (H_t, W_t, C_s*ps*ps)

    C_style = source_style.shape[2]
    ps_sq = patch_size * patch_size

    # Gather matched source patches
    H_s, W_s = source_style.shape[:2]
    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)

    matched_source_patches = source_style_patches[
        source_y, source_x
    ]  # (H_t, W_t, C_s*ps*ps)

    # Try Metal-accelerated NCC computation
    if (
        torch.backends.mps.is_available()
        and compiled_metal_ops is not None
        and source_style.device.type == "mps"
    ):
        # Flatten patches for Metal kernel
        matched_source_flat = (
            matched_source_patches.reshape(-1, C_style * ps_sq).float().contiguous()
        )
        target_style_flat = (
            target_style_patches.reshape(-1, C_style * ps_sq).float().contiguous()
        )

        # Ensure style_weights are contiguous
        style_weights_contiguous = style_weights.contiguous()

        # Call Metal NCC kernel
        style_error = compiled_metal_ops.mps_compute_patch_ncc(
            matched_source_flat, target_style_flat, style_weights_contiguous
        ).reshape(H_t, W_t)
    else:
        # Fallback to PyTorch implementation
        # Reshape to (H_t, W_t, C, ps*ps)
        matched_source = matched_source_patches.view(H_t, W_t, C_style, ps_sq)
        target = target_style_patches.view(H_t, W_t, C_style, ps_sq)

        # Average across channels for NCC computation
        s_vals = matched_source.float().mean(dim=2)  # (H_t, W_t, ps*ps)
        t_vals = target.float().mean(dim=2)  # (H_t, W_t, ps*ps)

        # Compute statistics
        mean_s = s_vals.mean(dim=2, keepdim=True)  # (H_t, W_t, 1)
        mean_t = t_vals.mean(dim=2, keepdim=True)

        std_s = s_vals.std(dim=2, keepdim=True, unbiased=False) + epsilon
        std_t = t_vals.std(dim=2, keepdim=True, unbiased=False) + epsilon

        # Covariance
        cov = ((s_vals - mean_s) * (t_vals - mean_t)).mean(dim=2)  # (H_t, W_t)

        # NCC and style error
        ncc = cov / (std_s.squeeze(2) * std_t.squeeze(2))
        style_error = (1.0 - ncc) * style_weights[0] * float(ps_sq)

    # Add guide SSD (reuse SSD function)
    if source_guide.shape[2] > 0:
        source_guide_patches = extract_patches(source_guide, patch_size)
        target_guide_patches = extract_patches(target_guide, patch_size)
        guide_error = compute_patch_ssd_vectorized(
            source_guide_patches, target_guide_patches, nnf, guide_weights
        )
        return style_error + guide_error
    else:
        return style_error
