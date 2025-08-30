from pathlib import Path
from typing import List, Optional

import numpy as np

from ezsynth.utils import io_utils
from ezsynth.utils.image_utils import resize_image_to_match

from .config import ProjectConfig


class ProjectData:
    """
    Manages all data loading, validation, and saving for an Ezsynth project.
    Acts as the single source of truth for all file I/O operations.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config

        # Define and create necessary directories
        self.content_dir = Path(config.content_dir)
        if isinstance(config.style_path, str):
            self.style_paths = [Path(config.style_path)]
        else:
            self.style_paths = [Path(p) for p in config.style_path]
        self.output_dir = Path(config.output_dir)
        self.cache_dir = Path(config.cache_dir)
        self.mask_dir = Path(config.mask_dir) if config.mask_dir else None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("Data Manager Initialized:")
        print(f"  - Content: {self.content_dir}")
        print(f"  - Style: {self.style_paths}")
        print(f"  - Output: {self.output_dir}")
        print(f"  - Cache: {self.cache_dir}")
        if self.mask_dir:
            print(f"  - Masks: {self.mask_dir}")

        self._content_frames: Optional[List[np.ndarray]] = None
        self._style_frames: Optional[List[np.ndarray]] = None
        self._mask_frames: Optional[List[np.ndarray]] = None

    def get_content_frames(self, force_reload: bool = False) -> List[np.ndarray]:
        """Loads, validates, and caches the content frames from disk."""
        if self._content_frames is None or force_reload:
            frames = io_utils.load_frames_from_dir(self.content_dir)
            if not frames:
                raise ValueError(f"No content frames found in {self.content_dir}")

            # Validate that all frames have the same resolution
            first_frame_shape = frames[0].shape
            for i, frame in enumerate(frames[1:]):
                if frame.shape != first_frame_shape:
                    raise ValueError(
                        f"Content frame resolution mismatch. Frame 0 is {first_frame_shape[:2]}, "
                        f"but frame {i+1} is {frame.shape[:2]}. All content frames must be the same size."
                    )
            self._content_frames = frames
        return self._content_frames

    def get_style_frames(self, force_reload: bool = False) -> List[np.ndarray]:
        """Loads and caches the style frames, optionally resizing them to match content."""
        if self._style_frames is None or force_reload:
            raw_styles = [io_utils.read_image(p) for p in self.style_paths]

            if self.config.force_style_size:
                print("`force_style_size` is enabled. Resizing styles...")
                content_frames = self.get_content_frames()
                if not content_frames:
                    raise ValueError(
                        "Cannot resize style frames without content frames to reference."
                    )
                reference_frame = content_frames[0]

                self._style_frames = [
                    resize_image_to_match(style, reference_frame)
                    for style in raw_styles
                ]
            else:
                self._style_frames = raw_styles

        return self._style_frames

    def get_mask_frames(self, force_reload: bool = False) -> Optional[List[np.ndarray]]:
        """Loads and caches the mask frames if a mask directory is specified."""
        if self.mask_dir:
            if self._mask_frames is None or force_reload:
                self._mask_frames = io_utils.load_masks_from_dir(self.mask_dir)
            return self._mask_frames
        return None

    def save_output_frames(self, frames: list[np.ndarray]):
        """Saves the final stylized frames to the output directory."""
        print(f"Saving {len(frames)} frames to {self.output_dir}...")
        for i, frame in enumerate(frames):
            filename = self.output_dir / f"{i:05d}.png"
            io_utils.write_image(filename, frame)
        print("All frames saved successfully.")
