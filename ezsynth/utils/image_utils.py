from typing import Sequence

import cv2
import numpy as np


def assert_uniform_image_shapes(images: Sequence[np.ndarray]) -> None:
    """Raise if ``images`` do not all share the same shape."""
    if len(images) < 2:
        return

    first_frame_shape = images[0].shape
    for i, image in enumerate(images[1:], start=1):
        if image.shape != first_frame_shape:
            raise ValueError(
                f"Content frame resolution mismatch. Frame 0 is {first_frame_shape[:2]}, "
                f"but frame {i} is {image.shape[:2]}. All content frames must be the same size."
            )


def resize_image_to_match(
    image_to_resize: np.ndarray, reference_image: np.ndarray
) -> np.ndarray:
    """
    Resizes an image to match the height and width of a reference image.

    Args:
        image_to_resize: The image that needs resizing.
        reference_image: The image providing the target dimensions.

    Returns:
        The resized image.
    """
    target_height, target_width = reference_image.shape[:2]
    current_height, current_width = image_to_resize.shape[:2]

    if (current_height, current_width) == (target_height, target_width):
        return image_to_resize  # No resize needed

    print(
        f"Resizing image from ({current_width}x{current_height}) to ({target_width}x{target_height})."
    )

    return cv2.resize(
        image_to_resize, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
