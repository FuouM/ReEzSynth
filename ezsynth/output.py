"""Output policies shared by CLI, API, service, and future integrations."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .data import ProjectData


@dataclass
class OutputResult:
    frames: List[np.ndarray]
    output_dir: Optional[Path]
    saved: bool


class OutputManager:
    """Apply a save/no-save policy to synthesized frames."""

    def __init__(self, data: ProjectData) -> None:
        self.data = data

    def handle(
        self,
        frames: List[np.ndarray],
        *,
        save: bool,
        visible_output_dir: Optional[Path],
    ) -> OutputResult:
        if save:
            self.data.save_output_frames(frames)
        return OutputResult(
            frames=frames,
            output_dir=visible_output_dir if save else None,
            saved=save,
        )
