# ezsynth/engines/edge_engine.py
from typing import List

import numpy as np
from tqdm import tqdm

# Refactored to import from its new local module
from ..edge.edge_detection import EdgeDetector
from .base import BaseEngine


class EdgeEngine(BaseEngine):
    """
    An engine responsible for computing edge maps for a sequence of frames.
    Wraps the underlying EdgeDetector implementation.
    """

    def __init__(self, method: str = "Classic"):
        """
        Initializes the EdgeEngine.

        Args:
            method (str): The edge detection algorithm to use ('Classic', 'PAGE', 'PST').
        """
        print(f"Initializing Edge Engine (method: {method})...")
        self.edge_detector = EdgeDetector(method=method)
        print("Edge Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes an edge map for each frame in the input sequence.

        Args:
            frames (List[np.ndarray]): A list of BGR frames.

        Returns:
            List[np.ndarray]: A list of corresponding BGR edge maps.
        """
        edge_maps = []
        for frame in tqdm(frames, desc="Computing Edge Maps"):
            edge_map = self.edge_detector.compute_edge(frame)
            # Ensure the output is a 3-channel BGR image for Ebsynth
            if len(edge_map.shape) == 2:
                edge_map = np.stack([edge_map] * 3, axis=-1)
            edge_maps.append(edge_map)
        return edge_maps
