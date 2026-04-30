import datetime
import platform
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F
from PIL import Image

PathLike = Union[str, Path]


def to_path(path: PathLike) -> str:
    return str(Path(path))


def load_precomputed(paths: dict[str, Path]) -> dict[str, str] | None:
    required = [paths["landmarks"], paths["pos"], paths["app"], paths["lut"]]
    try:
        if all(p.exists() and p.stat().st_size > 0 for p in required):
            return {
                "landmarks_path": to_path(paths["landmarks"]),
                "style_pos_guide_path": to_path(paths["pos"]),
                "style_app_guide_path": to_path(paths["app"]),
                "lut_path": to_path(paths["lut"]),
            }
    except OSError:
        return None
    return None


def read_image_pil(path: PathLike, convert="RGB") -> np.ndarray:
    """Read image using PIL and return as BGR numpy array (OpenCV format)."""
    img = Image.open(to_path(path)).convert(convert)
    arr = np.array(img)
    # Convert RGB to BGR for consistency with OpenCV
    return arr[..., ::-1].copy()


def normalize_landmarks(landmarks: Sequence[Tuple[int, int]]) -> list[Tuple[int, int]]:
    return [(int(x), int(y)) for x, y in landmarks]


def clamp_landmarks(
    landmarks: Sequence[Tuple[int, int]], size: Tuple[int, int]
) -> list[Tuple[int, int]]:
    h, w = size
    clamped = []
    for x, y in landmarks:
        clamped.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    return clamped


def write_image_pil(path: PathLike, image: np.ndarray) -> None:
    """Write BGR numpy array as image using PIL."""
    # Convert BGR to RGB
    rgb = image[..., ::-1]
    img = Image.fromarray(rgb)
    img.save(to_path(path))


def style_asset_paths(
    out_dir: Path, stem: str, lut_path: Path | None
) -> dict[str, Path]:
    default_lut_path = out_dir / f"{stem}_lut.bytes"
    lut_candidates: list[Path] = []
    if lut_path is not None:
        lut_candidates.append(lut_path)
    lut_candidates.append(default_lut_path)

    resolved_lut_path = lut_candidates[0]
    for candidate in lut_candidates:
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                resolved_lut_path = candidate
                break
        except OSError:
            continue

    return {
        "landmarks": out_dir / f"{stem}_landmarks.txt",
        "pos": out_dir / f"{stem}_style_pos.png",
        "app": out_dir / f"{stem}_style_app.png",
        "lut": resolved_lut_path,
    }


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale using PyTorch."""
    if image.ndim == 2:
        return to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 3:
        # Use PyTorch for grayscale conversion (matches cv2.cvtColor weights)
        # BGR format: [B, G, R] at indices [0, 1, 2]
        r = image[..., 2] * 0.299
        g = image[..., 1] * 0.587
        b = image[..., 0] * 0.114
        return to_uint8(r + g + b)
    raise ValueError("Expected grayscale or BGR image")


def read_landmarks_file(path: Path | str) -> list[Tuple[int, int]]:
    text = Path(path).read_text().strip().splitlines()
    pts: list[Tuple[int, int]] = []
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


def pack_lut_packed(
    lut: np.ndarray | torch.Tensor,
    *,
    cost: int | np.ndarray | torch.Tensor = 0,
) -> np.ndarray:
    """Inverse of :func:`unpack_lut_packed`: ``(..., 2)`` uint16 ``[sx, sy]`` → int32 packed word.

    Layout matches Taichi: ``(cost & 0x7FF) << 20 | (sx & 0x3FF) << 10 | (sy & 0x3FF)``.
    On-disk / PyTorch LUTs omit cost; use ``cost=0`` (default) unless you have the original cost field.
    """
    if isinstance(lut, torch.Tensor):
        lut_np = lut.detach().cpu().numpy()
    else:
        lut_np = np.asarray(lut)
    if lut_np.ndim != 4 or lut_np.shape[-1] != 2:
        raise ValueError(f"Expected LUT shape (*, *, *, 2), got {lut_np.shape}")
    sx = lut_np[..., 0].astype(np.int32, copy=False) & 0x3FF
    sy = lut_np[..., 1].astype(np.int32, copy=False) & 0x3FF
    if isinstance(cost, torch.Tensor):
        c = cost.detach().cpu().numpy()
    else:
        c = np.asarray(cost, dtype=np.int32)
    c = np.bitwise_and(c, 0x7FF)
    return (np.left_shift(c, 20) | np.left_shift(sx, 10) | sy).astype(np.int32)


def load_look_up_cube(path: Path | str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = 256 * 256 * 256 * 2
    if data.size != expected:
        raise ValueError(f"Unexpected LUT size {data.size}, expected {expected}")
    return data.reshape((256, 256, 256, 2))


def get_timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S")


def bgr_to_yuv(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to YUV using PyTorch (matches cv2.cvtColor)."""
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


def to_numpy_hwc(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_torch_device(device=None):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)
    return device


def get_taichi_arch(device: str | None = None):
    """Pick Taichi arch. When ``device`` is None, use the same auto-defaults as before."""
    if device is not None:
        d = device.lower().split(":", maxsplit=1)[0]
        if d == "cpu":
            return ti.cpu
        if d == "cuda":
            return ti.cuda if torch.cuda.is_available() else ti.cpu
        if d == "mps":
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return ti.metal
            return ti.cpu
        return ti.cpu

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return ti.metal
    elif torch.cuda.is_available():
        return ti.cuda
    return ti.cpu


def resize_torch(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image using PyTorch. size is (width, height)."""
    # Convert to tensor (H, W, C) -> (1, C, H, W)
    if image.ndim == 2:
        t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        t_resized = F.interpolate(
            t, size=(size[1], size[0]), mode="bilinear", align_corners=False
        )
        return t_resized.squeeze(0).squeeze(0).numpy().astype(image.dtype)
    else:
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        t_resized = F.interpolate(
            t, size=(size[1], size[0]), mode="bilinear", align_corners=False
        )
        return t_resized.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)
