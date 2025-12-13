from typing import Callable, List, Optional

import numpy as np

from .balanced_runner import BalancedModeRunner
from .config import FastBlendConfig


class FastBlendRunner:
    def __init__(self, config: Optional[FastBlendConfig] = None):
        """
        Initialize FastBlendRunner with configuration.

        Args:
            config: FastBlendConfig instance. If None, uses default balanced config.
        """
        self.config = config or FastBlendConfig()

    def run(
        self,
        frames_guide: List[np.ndarray],
        frames_style: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        backend: str = "auto",
    ) -> List[np.ndarray]:
        """
        Runs the FastBlend algorithm using the CuPy-based BalancedModeRunner.

        Args:
            frames_guide: Content frames (uint8)
            frames_style: Stylized frames (uint8)
            progress_callback: Optional callback function for progress updates (current, total)
            backend: Backend to use ("auto", "cuda", "cupy")

        Returns:
            List of processed frames (uint8)
        """
        if not self.config.enabled:
            return frames_style

        print(f"Starting FastBlend (Accuracy={self.config.accuracy})...")

        # Use backend from config if set to "auto", otherwise use parameter
        actual_backend = self.config.backend if backend == "auto" else backend

        # Create ebsynth config for the BalancedModeRunner
        ebsynth_config = self.config.get_ebsynth_config()

        # Convert frames to float32 as expected by the BalancedModeRunner
        frames_guide_float = [frame.astype(np.float32) for frame in frames_guide]
        frames_style_float = [frame.astype(np.float32) for frame in frames_style]

        # Use the BalancedModeRunner with progress tracking
        balanced_runner = BalancedModeRunner()
        result_frames = balanced_runner.run(
            frames_guide_float,
            frames_style_float,
            batch_size=self.config.batch_size,
            window_size=self.config.window_size,
            ebsynth_config=ebsynth_config,
            desc="FastBlend Processing",
            progress_callback=progress_callback,
            backend=actual_backend,
        )

        # Convert back to uint8
        final_frames = [
            np.clip(frame, 0, 255).astype(np.uint8) for frame in result_frames
        ]

        print("FastBlend complete.")
        return final_frames
