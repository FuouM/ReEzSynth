from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def np_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """Converts a HxWxC numpy image to a BxCxHxW torch tensor."""
    if len(img_np.shape) == 3:
        img_np = np.expand_dims(img_np, 0)  # Add batch dimension if missing
    return torch.from_numpy(img_np).permute(0, 3, 1, 2).float() / 255.0


def tensor_to_np(img_tensor: torch.Tensor) -> np.ndarray:
    """Converts a BxCxHxW torch tensor to a HxWxC numpy image."""
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_laplacian_pyramid(frame: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Constructs a Laplacian pyramid from a single numpy frame.
    """
    pyramid = []
    current_tensor = np_to_tensor(frame)

    for _ in range(levels - 1):
        if current_tensor.size(2) < 2 or current_tensor.size(3) < 2:
            break  # Stop if image is too small

        h, w = current_tensor.size(2), current_tensor.size(3)
        downsampled = F.interpolate(
            current_tensor, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        upsampled = F.interpolate(
            downsampled, size=(h, w), mode="bilinear", align_corners=False
        )

        laplacian_level = current_tensor - upsampled
        pyramid.append(tensor_to_np(laplacian_level))

        current_tensor = downsampled

    pyramid.append(tensor_to_np(current_tensor))  # Add the coarsest level (base)
    return pyramid


def reconstruct_from_laplacian_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
    """
    Reconstructs an image from its Laplacian pyramid.
    """
    # Start with the coarsest level
    reconstructed_tensor = np_to_tensor(pyramid[-1])

    # Iterate from second-to-last level up to the finest
    for i in range(len(pyramid) - 2, -1, -1):
        laplacian_level = np_to_tensor(pyramid[i])
        h, w = laplacian_level.size(2), laplacian_level.size(3)

        upsampled = F.interpolate(
            reconstructed_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        reconstructed_tensor = upsampled + laplacian_level

    return tensor_to_np(reconstructed_tensor)


def downsample_image(frame: np.ndarray, level: int) -> np.ndarray:
    """Downsamples an image by a factor of 2^level."""
    if level == 0:
        return frame

    tensor = np_to_tensor(frame)
    downsampled = F.interpolate(
        tensor, scale_factor=(1 / (2**level)), mode="bilinear", align_corners=False
    )
    return tensor_to_np(downsampled)


def downsample_flow(flow: np.ndarray, level: int) -> np.ndarray:
    """Downsamples a flow field, scaling the vectors appropriately."""
    if level == 0:
        return flow

    h, w, _ = flow.shape
    new_h, new_w = h // (2**level), w // (2**level)

    # Use torch for resizing
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
    downsampled_flow = F.interpolate(
        flow_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Scale flow vectors
    downsampled_flow /= 2**level

    return downsampled_flow.squeeze(0).permute(1, 2, 0).numpy()
