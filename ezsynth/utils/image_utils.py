# ezsynth/utils/image_utils.py
import cv2
import numpy as np


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
