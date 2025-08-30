from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def np_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """Converts a HxWxC numpy image to a BxCxHxW torch tensor in float[0,1] range."""
    if len(img_np.shape) == 3:
        img_np = np.expand_dims(img_np, 0)
    # Ensure input is float before permuting
    img_float = img_np.astype(np.float32)
    return torch.from_numpy(img_float).permute(0, 3, 1, 2) / 255.0


def tensor_to_np(img_tensor: torch.Tensor) -> np.ndarray:
    """Converts a BxCxHxW torch tensor back to a HxWxC uint8 numpy image."""
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_laplacian_pyramid(frame: np.ndarray, levels: int) -> List[torch.Tensor]:
    """
    Constructs a Laplacian pyramid from a single numpy frame.
    Returns a list of torch.Tensors. The detail layers are float32 centered at 0.
    """
    pyramid_tensors = []
    current_tensor = np_to_tensor(frame)

    for _ in range(levels - 1):
        h, w = current_tensor.size(2), current_tensor.size(3)
        downsampled = F.interpolate(
            current_tensor, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        upsampled = F.interpolate(
            downsampled, size=(h, w), mode="bilinear", align_corners=False
        )

        laplacian_level = current_tensor - upsampled
        pyramid_tensors.append(laplacian_level)

        current_tensor = downsampled

    pyramid_tensors.append(current_tensor)  # Add the coarsest level (base)
    return pyramid_tensors


def reconstruct_from_laplacian_pyramid(
    pyramid_tensors: List[torch.Tensor],
) -> np.ndarray:
    """
    Reconstructs an image from its Laplacian pyramid of torch.Tensors.
    """
    reconstructed_tensor = pyramid_tensors[-1]

    for i in range(len(pyramid_tensors) - 2, -1, -1):
        laplacian_level = pyramid_tensors[i]
        h, w = laplacian_level.size(2), laplacian_level.size(3)

        upsampled = F.interpolate(
            reconstructed_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        reconstructed_tensor = upsampled + laplacian_level

    return tensor_to_np(reconstructed_tensor)


def remap_residual_to_image(residual_tensor: torch.Tensor) -> np.ndarray:
    """
    Remaps a float tensor residual (centered at 0) to a uint8 numpy image (centered at 128).
    """
    # Shift from [-1, 1] range to [0, 2] range, then scale to [0, 255]
    image_tensor = (residual_tensor + 1.0) / 2.0
    return tensor_to_np(image_tensor)


def remap_image_to_residual(image: np.ndarray) -> torch.Tensor:
    """
    Remaps a uint8 numpy image back to a float tensor residual.
    """
    # Convert from [0, 255] uint8 to [0, 1] float tensor
    image_tensor = np_to_tensor(image)
    # Shift from [0, 1] range to [-0.5, 0.5] range for residual
    # NOTE: The exact scaling might need tuning, but this is a robust start.
    residual_tensor = (image_tensor - 0.5) * 2.0
    return residual_tensor


def downsample_image(frame: np.ndarray, level: int) -> np.ndarray:
    if level == 0:
        return frame
    downsampled = frame
    for _ in range(level):
        downsampled = cv2.pyrDown(downsampled)
    return downsampled


def downsample_flow(flow: np.ndarray, level: int) -> np.ndarray:
    if level == 0:
        return flow
    h, w, _ = flow.shape
    new_h, new_w = h // (2**level), w // (2**level)
    downsampled_flow = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    downsampled_flow /= 2**level
    return downsampled_flow
