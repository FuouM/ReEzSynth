# ezsynth/utils/io_utils.py
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _decode(path: str | Path, flags: int) -> np.ndarray:
    # np.fromfile + imdecode instead of cv2.imread: handles non-ASCII Windows paths.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise OSError(f"Could not decode image at {path}")
    return image


def read_image(path: str | Path) -> np.ndarray:
    """Reads an image as 8-bit 3-channel BGR (alpha dropped, gray expanded).

    EXIF orientation is ignored so pixels match the stored frame layout.
    """
    try:
        return _decode(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except Exception as e:
        raise OSError(f"Error reading image at {path}: {e}") from e


def read_mask(path: str | Path) -> np.ndarray:
    """Reads a mask as a single-channel grayscale image."""
    try:
        return _decode(path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise OSError(f"Error reading mask at {path}: {e}") from e


def write_image(path: str | Path, image: np.ndarray):
    """Writes a BGR (or grayscale) image; the format follows the file extension."""
    try:
        ok, encoded = cv2.imencode(Path(path).suffix or ".png", image)
        if not ok:
            raise OSError("encoding failed")
        encoded.tofile(str(path))
    except Exception as e:
        raise OSError(f"Error writing image to {path}: {e}") from e


def get_sorted_image_paths(dir_path: Path) -> list[Path]:
    """Gets all image paths from a directory, sorted naturally."""
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    paths = [p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

    if not paths:
        raise FileNotFoundError(f"No image files found in {dir_path}")

    # Natural sort
    paths.sort(
        key=lambda x: [
            int(c) if c.isdigit() else c for c in re.split("([0-9]+)", x.stem)
        ]
    )
    return paths


def load_frames_from_dir(dir_path: Path) -> list[np.ndarray]:
    """Loads a sequence of frames from a directory."""
    paths = get_sorted_image_paths(dir_path)
    frames = [
        read_image(p) for p in tqdm(paths, desc=f"Loading frames from {dir_path.name}")
    ]
    return frames


def load_masks_from_dir(dir_path: Path) -> list[np.ndarray]:
    """Loads a sequence of masks from a directory."""
    paths = get_sorted_image_paths(dir_path)
    masks = [
        read_mask(p) for p in tqdm(paths, desc=f"Loading masks from {dir_path.name}")
    ]
    return masks
