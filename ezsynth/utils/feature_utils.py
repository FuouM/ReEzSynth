# ezsynth/utils/feature_utils.py
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def generate_tracked_features(
    initial_frame: np.ndarray,
    flows: List[np.ndarray],
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: int = 10,
) -> List[np.ndarray]:
    """
    Finds good features to track in the first frame and propagates them
    through the sequence using optical flow.

    Args:
        initial_frame (np.ndarray): The first frame of the content sequence.
        flows (List[np.ndarray]): The list of forward optical flows (i -> i+1).
        max_corners (int): Maximum number of features to detect.
        quality_level (float): Quality level for corner detection.
        min_distance (int): Minimum distance between detected features.

    Returns:
        List[np.ndarray]: A list where each element is an array of feature
                          coordinates for that frame.
    """
    # 1. Detect initial features in the first frame
    gray_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    initial_points = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )

    if initial_points is None:
        print("Warning: No features found to track.")
        # Return a list of empty arrays matching the expected number of frames
        return [np.array([]) for _ in range(len(flows) + 1)]

    # Reshape points from [[x, y]] to [x, y]
    current_points = initial_points.reshape(-1, 2)
    tracked_points_over_time = [current_points]

    # 2. Track features frame-by-frame using the pre-computed flow
    for flow_field in tqdm(flows, desc="Tracking Sparse Features"):
        h, w, _ = flow_field.shape
        next_points = []
        for point in current_points:
            x, y = point
            # Ensure point is within bounds before looking up flow
            if 0 <= x < w and 0 <= y < h:
                # Get flow vector at the point's location
                flow_vector = flow_field[int(y), int(x)]
                # Update point position
                new_point = point + flow_vector
                next_points.append(new_point)
            else:
                # If point goes out of bounds, just append its last known position
                next_points.append(point)

        current_points = np.array(next_points)
        tracked_points_over_time.append(current_points)

    return tracked_points_over_time


def render_gaussian_guide(
    height: int, width: int, points: np.ndarray, radius: int = 3, blur_ksize: int = 15
) -> np.ndarray:
    """
    Renders a set of points as soft, blurred dots on a guide image.

    Args:
        height (int): The height of the guide image.
        width (int): The width of the guide image.
        points (np.ndarray): An array of (x, y) coordinates.
        radius (int): The radius of the circles to draw for each point.
        blur_ksize (int): The kernel size for the Gaussian blur (must be odd).

    Returns:
        np.ndarray: A 3-channel BGR guide image.
    """
    guide = np.zeros((height, width), dtype=np.uint8)

    if points.size == 0:
        return np.stack([guide] * 3, axis=-1)

    for point in points:
        center = tuple(np.round(point).astype(int))
        cv2.circle(guide, center, radius, 255, -1)

    # Apply blur to make the points "soft"
    guide = cv2.GaussianBlur(guide, (blur_ksize, blur_ksize), 0)

    # Convert to 3-channel BGR for Ebsynth
    return np.stack([guide] * 3, axis=-1)
