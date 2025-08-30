# ezsynth/utils/mask_utils.py
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies a mask to a single image."""
    return cv2.bitwise_and(image, image, mask=mask)


def apply_masks_to_sequence(
    images: List[np.ndarray], masks: List[np.ndarray]
) -> List[np.ndarray]:
    """Applies a sequence of masks to a sequence of images."""
    if len(images) != len(masks):
        raise ValueError(
            f"Image sequence length ({len(images)}) and mask sequence length ({len(masks)}) must match."
        )

    masked_images = []
    for img, msk in zip(images, masks):
        masked_images.append(apply_mask(img, msk))
    return masked_images


def composite_masked_image(
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 0,
) -> np.ndarray:
    """Composites a processed (masked) region back onto the original image."""
    if feather_radius > 0:
        # Ensure feather_radius is odd
        radius = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        mask_blurred = cv2.GaussianBlur(mask, (radius, radius), 0)
        mask_float = mask_blurred.astype(np.float32) / 255.0
    else:
        mask_float = mask.astype(np.float32) / 255.0

    # Ensure mask is 3-channel for broadcasting
    if len(mask_float.shape) == 2:
        mask_float = np.expand_dims(mask_float, axis=-1)

    background = original.astype(np.float32) * (1.0 - mask_float)
    foreground = processed.astype(np.float32) * mask_float

    result = (background + foreground).clip(0, 255).astype(np.uint8)
    return result


def composite_sequence(
    originals: List[np.ndarray],
    processeds: List[np.ndarray],
    masks: List[np.ndarray],
    feather: int = 0,
) -> List[np.ndarray]:
    """Composites a sequence of processed images back onto their originals."""
    if not (len(originals) == len(processeds) == len(masks)):
        raise ValueError("All sequence lengths must match for compositing.")

    composited_frames = []
    for orig, proc, msk in tqdm(
        zip(originals, processeds, masks),
        total=len(originals),
        desc="Compositing masked frames",
    ):
        composited_frames.append(composite_masked_image(orig, proc, msk, feather))

    return composited_frames
