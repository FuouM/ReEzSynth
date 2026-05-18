from typing import List

import numpy as np

from ..edge.batch import compute_edge


class EdgeEngine:
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
        self.method = method
        print("Edge Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes an edge map for each frame in the input sequence.

        Args:
            frames (List[np.ndarray]): A list of BGR frames.

        Returns:
            List[np.ndarray]: A list of corresponding BGR edge maps.
        """
        return compute_edge(frames, self.method)
