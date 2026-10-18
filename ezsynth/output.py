"""Output policies shared by CLI, API, service, and future integrations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ProjectData


@dataclass
class OutputResult:
    frames: list[np.ndarray]
    output_dir: Path | None
    saved: bool


class OutputManager:
    """Apply a save/no-save policy to synthesized frames."""

    def __init__(self, data: ProjectData) -> None:
        self.data = data

    def handle(
        self,
        frames: list[np.ndarray],
        *,
        save: bool,
        visible_output_dir: Path | None,
    ) -> OutputResult:
        if save:
            self.data.save_output_frames(frames)
        return OutputResult(
            frames=frames,
            output_dir=visible_output_dir if save else None,
            saved=save,
        )
