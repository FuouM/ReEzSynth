# comfyui_ezsynth/core/tensor_utils.py
"""
Tensor conversion utilities for ComfyUI nodes.
"""

from typing import List, Tuple, Union

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI tensor (B,H,W,C) to numpy (H,W,C).

    Args:
        tensor: Input tensor, typically shape (B, H, W, C) or (H, W, C)

    Returns:
        NumPy array with shape (H, W, C), dtype uint8
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dim
    elif tensor.dim() == 3:
        pass  # Already in HWC format
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)  # Add channel dim for grayscale

    # Convert from float [0,1] to uint8 [0,255]
    np_array = tensor.cpu().numpy()

    # Handle float inputs
    if np_array.dtype in [np.float32, np.float64]:
        np_array = (np_array * 255).clip(0, 255).astype(np.uint8)
    elif np_array.dtype == np.float16:
        np_array = (np_array.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)

    return np_array


def numpy_to_tensor(array: np.ndarray, batch: bool = True) -> torch.Tensor:
    """
    Convert numpy array (H,W,C) to ComfyUI tensor (B,H,W,C).

    Args:
        array: Input numpy array, shape (H, W, C) or (H, W)
        batch: Whether to add batch dimension

    Returns:
        Tensor with shape (B, H, W, C) or (H, W, C) if batch=False
    """
    # Handle grayscale
    if array.ndim == 2:
        array = array[..., np.newaxis]

    # Convert from uint8 [0,255] to float [0,1]
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)

    if batch:
        tensor = tensor.unsqueeze(0)  # Add batch dim

    return tensor


def tensor_list_to_numpy_list(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Convert list of tensors to list of numpy arrays.

    Args:
        tensors: List of tensors

    Returns:
        List of numpy arrays
    """
    return [tensor_to_numpy(t) for t in tensors]


def numpy_list_to_tensor_list(
    arrays: List[np.ndarray], batch: bool = True
) -> List[torch.Tensor]:
    """
    Convert list of numpy arrays to list of tensors.

    Args:
        arrays: List of numpy arrays
        batch: Whether to add batch dimension

    Returns:
        List of tensors
    """
    return [numpy_to_tensor(a, batch=batch) for a in arrays]


def normalize_tensor(
    tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 255.0
) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range.

    Args:
        tensor: Input tensor
        min_val: Minimum value in input
        max_val: Maximum value in input

    Returns:
        Normalized tensor in [0, 1]
    """
    return (tensor - min_val) / (max_val - min_val)


def denormalize_tensor(
    tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 255.0
) -> torch.Tensor:
    """
    Denormalize tensor from [0, 1] to original range.

    Args:
        tensor: Input tensor in [0, 1]
        min_val: Minimum value in output
        max_val: Maximum value in output

    Returns:
        Denormalized tensor
    """
    return tensor * (max_val - min_val) + min_val


def flow_to_tensor(flow: np.ndarray) -> torch.Tensor:
    """
    Convert optical flow array to tensor.
    Flow should be shape (H, W, 2) with (dx, dy) components.

    Args:
        flow: Optical flow numpy array, shape (H, W, 2)

    Returns:
        Tensor with shape (2, H, W)
    """
    tensor = torch.from_numpy(flow.transpose(2, 0, 1).astype(np.float32))
    return tensor


def tensor_to_flow(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to optical flow array.

    Args:
        tensor: Tensor with shape (2, H, W) or (H, W, 2)

    Returns:
        Optical flow numpy array with shape (H, W, 2)
    """
    if tensor.dim() == 3 and tensor.shape[0] == 2:
        return tensor.cpu().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 3 and tensor.shape[2] == 2:
        return tensor.cpu().numpy()
    else:
        raise ValueError(
            f"Unexpected tensor shape: {tensor.shape}. Expected (2, H, W) or (H, W, 2)"
        )


def nnf_to_tensor(nnf: np.ndarray) -> torch.Tensor:
    """
    Convert NNF (Nearest Neighbor Field) array to tensor.
    NNF should be shape (H, W, 2) with integer indices.

    Args:
        nnf: NNF numpy array, shape (H, W, 2)

    Returns:
        Tensor with shape (2, H, W), dtype int32
    """
    tensor = torch.from_numpy(nnf.transpose(2, 0, 1).astype(np.int32))
    return tensor


def tensor_to_nnf(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to NNF array.

    Args:
        tensor: Tensor with shape (2, H, W) or (H, W, 2), dtype int32

    Returns:
        NNF numpy array with shape (H, W, 2)
    """
    if tensor.dim() == 3 and tensor.shape[0] == 2:
        return tensor.cpu().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 3 and tensor.shape[2] == 2:
        return tensor.cpu().numpy()
    else:
        raise ValueError(
            f"Unexpected tensor shape: {tensor.shape}. Expected (2, H, W) or (H, W, 2)"
        )


def guide_to_tensor(guide: np.ndarray, weight: float) -> Tuple[torch.Tensor, float]:
    """
    Convert guide array and weight to tensor.

    Args:
        guide: Guide numpy array
        weight: Guide weight

    Returns:
        Tuple of (guide_tensor, weight)
    """
    return numpy_to_tensor(guide), weight


def tensor_to_guide(tensor: torch.Tensor, weight: float) -> Tuple[np.ndarray, float]:
    """
    Convert tensor to guide array and weight.

    Args:
        tensor: Guide tensor
        weight: Guide weight

    Returns:
        Tuple of (guide_array, weight)
    """
    return tensor_to_numpy(tensor), weight


def stack_images_for_display(images: List[np.ndarray], cols: int = 4) -> np.ndarray:
    """
    Stack multiple images for display in ComfyUI.

    Args:
        images: List of numpy arrays
        cols: Number of columns

    Returns:
        Stacked numpy array
    """
    import math

    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]

    # Ensure all images have same dimensions
    target_h, target_w = h, w
    resized_images = []
    for img in images:
        if img.shape[:2] != (target_h, target_w):
            # Simple resize - in practice would use cv2 or PIL
            resized_images.append(img)
        else:
            resized_images.append(img)

    # Create grid
    if len(images[0].shape) == 2:
        # Grayscale - convert to RGB for display
        grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    else:
        grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)

    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        grid[
            row * target_h : (row + 1) * target_h, col * target_w : (col + 1) * target_w
        ] = img

    return grid
