# ezsynth/torch_ops/mask_ops.py
"""
Mask operations for convergence checking in EBSynth.

These operations create and dilate masks that identify regions that have
changed significantly between iterations, enabling early termination.
"""

import torch
import torch.nn.functional as F


def evaluate_mask(
    style1: torch.Tensor,  # (H_t, W_t, C) uint8 - current iteration
    style2: torch.Tensor,  # (H_t, W_t, C) uint8 - previous iteration
    stop_threshold: int,
) -> torch.Tensor:
    """
    Create a binary mask of pixels that have changed significantly.

    Args:
        style1: Current synthesized image
        style2: Previous synthesized image
        stop_threshold: Minimum difference to consider "changed"

    Returns:
        mask: (H_t, W_t) uint8 tensor where 255 = changed, 0 = converged

    Implementation:
    - Compute absolute difference across all channels
    - Take max difference per pixel
    - Threshold to create binary mask
    """
    # Compute absolute difference: (H, W, C)
    diff = (style1.int() - style2.int()).abs()

    # Max difference across channels: (H, W)
    max_diff = diff.max(dim=2).values

    # Threshold to create binary mask
    mask = (max_diff >= stop_threshold).to(torch.uint8) * 255

    return mask


def dilate_mask(
    mask_in: torch.Tensor,  # (H_t, W_t) uint8
    patch_size: int,
) -> torch.Tensor:
    """
    Dilate the convergence mask by patch_size radius.

    This ensures that when we check convergence, we also consider the
    influence region of each patch.

    Args:
        mask_in: Binary mask (255 = active, 0 = converged)
        patch_size: Dilation radius (typically 7)

    Returns:
        mask_out: Dilated binary mask

    Implementation:
    - Use max_pool2d for efficient morphological dilation
    - Kernel size = patch_size, stride = 1, padding = patch_size//2
    """
    # Convert to float for max pooling: (1, 1, H, W)
    mask_float = mask_in.unsqueeze(0).unsqueeze(0).float()

    # Dilate using max pooling
    dilated = F.max_pool2d(
        mask_float, kernel_size=patch_size, stride=1, padding=patch_size // 2
    )

    # Convert back to uint8
    mask_out = dilated.squeeze(0).squeeze(0).to(torch.uint8)

    return mask_out
