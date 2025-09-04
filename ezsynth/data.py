from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ezsynth.utils import io_utils
from ezsynth.utils.image_utils import resize_image_to_match

from .config import ProjectConfig


def _save_frame_worker(args: tuple):
    """Helper function for multiprocessing pool to save a single frame."""
    i, frame, output_dir_str = args
    filename = Path(output_dir_str) / f"{i:05d}.png"
    io_utils.write_image(filename, frame)


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
        self.modulation_dir = (
            Path(config.modulation_dir) if config.modulation_dir else None
        )  # New

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("Data Manager Initialized:")
        print(f"  - Content: {self.content_dir}")
        print(f"  - Style: {self.style_paths}")
        print(f"  - Output: {self.output_dir}")
        print(f"  - Cache: {self.cache_dir}")
        if self.mask_dir:
            print(f"  - Masks: {self.mask_dir}")
        if self.modulation_dir:  # New
            print(f"  - Modulations: {self.modulation_dir}")

        self._content_frames: Optional[List[np.ndarray]] = None
        self._style_frames: Optional[List[np.ndarray]] = None
        self._mask_frames: Optional[List[np.ndarray]] = None
        self._modulation_frames: Optional[List[np.ndarray]] = None  # New

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

    def get_modulation_frames(
        self, force_reload: bool = False
    ) -> Optional[List[np.ndarray]]:
        """Loads and caches modulation frames if a directory is specified."""
        if self.modulation_dir:
            if self._modulation_frames is None or force_reload:
                # Modulation maps are loaded just like regular frames
                self._modulation_frames = io_utils.load_frames_from_dir(
                    self.modulation_dir
                )
            return self._modulation_frames
        return None

    def save_output_frames(self, frames: list[np.ndarray]):
        """Saves the final stylized frames to the output directory in parallel."""
        num_frames = len(frames)
        if num_frames == 0:
            print("No frames to save.")
            return

        print(
            f"Saving {num_frames} frames to {self.output_dir} using parallel processing..."
        )

        # Prepare arguments for the worker processes.
        # Passing the output directory as a string is safer for pickling.
        tasks = [(i, frame, str(self.output_dir)) for i, frame in enumerate(frames)]

        # Use a sensible number of processes to avoid I/O bottlenecks.
        # Capped at 12 workers, which is a reasonable upper limit for this task.
        num_workers = min(cpu_count(), 12, num_frames)

        with Pool(processes=num_workers) as pool:
            # Use tqdm to show a progress bar for the saving process.
            # imap_unordered is generally faster as it yields results as they complete.
            list(
                tqdm(
                    pool.imap_unordered(_save_frame_worker, tasks),
                    total=num_frames,
                    desc="Saving Output Frames",
                )
            )

        print("All frames saved successfully.")
