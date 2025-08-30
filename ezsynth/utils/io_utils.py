# ezsynth/utils/io_utils.py
import re
from pathlib import Path
from typing import List, Union

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Reads an image and converts it to BGR format."""
    try:
        img = imageio.imread(path)
        if img.shape[-1] == 4:  # Handle RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) == 2:  # Handle Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:  # Handle RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        raise IOError(f"Error reading image at {path}: {e}")


def read_mask(path: Union[str, Path]) -> np.ndarray:
    """Reads a mask as a single-channel grayscale image."""
    try:
        # IMREAD_GRAYSCALE ensures a 2D array
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask at {path}")
        return mask
    except Exception as e:
        raise IOError(f"Error reading mask at {path}: {e}")


def write_image(path: Union[str, Path], image: np.ndarray):
    """Writes an image in BGR format to disk."""
    try:
        # Convert back to RGB for standard image viewers
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(path, image_rgb)
    except Exception as e:
        raise IOError(f"Error writing image to {path}: {e}")


def get_sorted_image_paths(dir_path: Path) -> List[Path]:
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


def load_frames_from_dir(dir_path: Path) -> List[np.ndarray]:
    """Loads a sequence of frames from a directory."""
    paths = get_sorted_image_paths(dir_path)
    frames = [
        read_image(p) for p in tqdm(paths, desc=f"Loading frames from {dir_path.name}")
    ]
    return frames


def load_masks_from_dir(dir_path: Path) -> List[np.ndarray]:
    """Loads a sequence of masks from a directory."""
    paths = get_sorted_image_paths(dir_path)
    masks = [
        read_mask(p) for p in tqdm(paths, desc=f"Loading masks from {dir_path.name}")
    ]
    return masks
