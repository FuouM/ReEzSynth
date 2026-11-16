from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ezsynth.utils.gpu_memory import apply_gpu_memory_limit

PathLike = str | Path


def read_image_pil(path: PathLike, convert="RGB") -> np.ndarray:
    """Read image using PIL and return as BGR numpy array (OpenCV format)."""
    img = Image.open(path).convert(convert)
    arr = np.array(img)
    # Convert RGB to BGR for consistency with OpenCV
    return arr[..., ::-1].copy()


def normalize_landmarks(landmarks: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in landmarks]


def write_image_pil(path: PathLike, image: np.ndarray) -> None:
    """Write BGR numpy array as image using PIL."""
    # Convert BGR to RGB
    rgb = image[..., ::-1]
    img = Image.fromarray(rgb)
    img.save(path)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale (cv2.cvtColor weights)."""
    if image.ndim == 2:
        return to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 3:
        # BGR format: [B, G, R] at indices [0, 1, 2]
        r = image[..., 2] * 0.299
        g = image[..., 1] * 0.587
        b = image[..., 0] * 0.114
        return to_uint8(r + g + b)
    raise ValueError("Expected grayscale or BGR image")


def read_landmarks_file(path: Path | str) -> list[tuple[int, int]]:
    text = Path(path).read_text().strip().splitlines()
    pts: list[tuple[int, int]] = []
    start_idx = 1 if text and text[0].strip() == "68" else 0
    for line in text[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            pts.append((int(float(parts[0])), int(float(parts[1]))))
    return pts


def save_look_up_cube(lut: np.ndarray, path: Path | str) -> Path:
    arr = np.asarray(lut, dtype=np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return Path(path)


def unpack_lut_packed(packed: np.ndarray | torch.Tensor) -> np.ndarray:
    """Unpack Taichi packed LUT ``(cost << 20) | (sx << 10) | sy`` to ``(256,256,256,2)`` uint16 ``[sx, sy]``."""
    if isinstance(packed, torch.Tensor):
        packed_np = packed.detach().cpu().numpy().astype(np.int32, copy=False)
    else:
        packed_np = np.asarray(packed, dtype=np.int32)
    lut_col = ((packed_np >> 10) & 0x3FF).astype(np.uint16)
    lut_row = (packed_np & 0x3FF).astype(np.uint16)
    return np.stack([lut_col, lut_row], axis=-1)


def pack_lut_packed(lut: np.ndarray | torch.Tensor) -> np.ndarray:
    """Inverse of :func:`unpack_lut_packed`: ``(..., 2)`` uint16 ``[sx, sy]`` -> int32 packed word.

    Layout matches Taichi: ``(cost & 0x7FF) << 20 | (sx & 0x3FF) << 10 | (sy & 0x3FF)``;
    on-disk / PyTorch LUTs carry no cost, so the cost field is 0.
    """
    if isinstance(lut, torch.Tensor):
        lut_np = lut.detach().cpu().numpy()
    else:
        lut_np = np.asarray(lut)
    if lut_np.ndim != 4 or lut_np.shape[-1] != 2:
        raise ValueError(f"Expected LUT shape (*, *, *, 2), got {lut_np.shape}")
    sx = lut_np[..., 0].astype(np.int32, copy=False) & 0x3FF
    sy = lut_np[..., 1].astype(np.int32, copy=False) & 0x3FF
    return ((sx << 10) | sy).astype(np.int32)


def load_look_up_cube(path: Path | str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = 256 * 256 * 256 * 2
    if data.size != expected:
        raise ValueError(f"Unexpected LUT size {data.size}, expected {expected}")
    return data.reshape((256, 256, 256, 2))


def bgr_to_yuv(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to YUV (BT.601, matches cv2.cvtColor)."""
    # BGR to RGB first
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0

    # RGB to YUV conversion matrix (ITU-R BT.601)
    # Y = 0.299*R + 0.587*G + 0.114*B
    # U = -0.14713*R - 0.28886*G + 0.436*B + 128
    # V = 0.615*R - 0.51499*G - 0.10001*B + 128
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5

    yuv = np.stack([y, u, v], axis=-1)
    return (yuv * 255.0).astype(np.uint8)


def to_tensor(image: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    arr = torch.from_numpy(image.astype(np.float32))
    if arr.ndim == 3:
        # Assume BGR input (OpenCV convention), convert to RGB for PyTorch
        arr = arr[..., [2, 1, 0]]  # BGR to RGB
        arr = arr.permute(2, 0, 1)  # HWC -> CHW
    return arr.to(device) / 255.0


def to_tensor_simple(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1) * 255.0).permute(1, 2, 0).detach().cpu().numpy()
    # Convert RGB back to BGR for OpenCV compatibility
    if arr.ndim == 3:
        arr = arr[..., [2, 1, 0]]  # RGB to BGR
    return to_uint8(arr)


def get_torch_device(device=None):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)
    apply_gpu_memory_limit(device)
    return device
