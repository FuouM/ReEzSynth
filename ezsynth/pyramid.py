# ezsynth/pyramid.py
import cv2
import numpy as np


def downsample_image(frame: np.ndarray, level: int) -> np.ndarray:
    """Recursively downsamples an image by a given number of pyramid levels."""
    if level == 0:
        return frame
    downsampled = frame
    for _ in range(level):
        # pyrDown halves the dimensions
        downsampled = cv2.pyrDown(downsampled)
    return downsampled


def downsample_flow(flow: np.ndarray, level: int) -> np.ndarray:
    """Downsamples an optical flow field and scales its vectors."""
    if level == 0:
        return flow
    h, w, _ = flow.shape
    scale_factor = 2**level
    new_h, new_w = h // scale_factor, w // scale_factor

    downsampled_flow = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Scale the flow vectors to match the new resolution
    downsampled_flow /= scale_factor

    return downsampled_flow
