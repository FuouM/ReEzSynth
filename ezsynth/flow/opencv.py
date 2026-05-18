"""OpenCV dense optical flow (DIS, Farneback)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .types import OpenCvFlowMethod


def compute_opencv_flow_sequence(
    frames: List[np.ndarray],
    method: OpenCvFlowMethod,
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    if method == "DIS":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
    elif method != "FARNEBACK":
        raise NotImplementedError(
            f"Unknown OpenCV flow method {method!r}; expected DIS or FARNEBACK"
        )

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    optical_flows: List[np.ndarray] = []
    for i in tqdm(
        range(len(frames) - 1),
        desc=f"Computing Optical Flow (CV2 {method})",
    ):
        prev_gray = gray_frames[i]
        curr_gray = gray_frames[i + 1]

        if method == "DIS":
            flow = dis.calc(prev_gray, curr_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
        optical_flows.append(flow)

    return optical_flows
