import numpy as np
import torch

__all__ = [
    "_to_uint8",
    "_ensure_grayscale",
    "_to_tensor",
    "_to_numpy_image",
]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale using PyTorch."""
    if image.ndim == 2:
        return _to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 3:
        # Use PyTorch for grayscale conversion (matches cv2.cvtColor weights)
        # BGR format: [B, G, R] at indices [0, 1, 2]
        r = image[..., 2] * 0.299
        g = image[..., 1] * 0.587
        b = image[..., 0] * 0.114
        return _to_uint8(r + g + b)
    raise ValueError("Expected grayscale or BGR image")


def _to_tensor(image: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    arr = torch.from_numpy(image.astype(np.float32))
    if arr.ndim == 3:
        # Assume BGR input (OpenCV convention), convert to RGB for PyTorch
        arr = arr[..., [2, 1, 0]]  # BGR to RGB
        arr = arr.permute(2, 0, 1)  # HWC -> CHW
    return arr.to(device) / 255.0


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1) * 255.0).permute(1, 2, 0).detach().cpu().numpy()
    # Convert RGB back to BGR for OpenCV compatibility
    if arr.ndim == 3:
        arr = arr[..., [2, 1, 0]]  # RGB to BGR
    return _to_uint8(arr)
