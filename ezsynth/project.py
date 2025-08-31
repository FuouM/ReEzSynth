# ezsynth/project.py
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .config import MainConfig
from .data import ProjectData
from .pipeline import SynthesisPipeline


class Project:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        self.config = self._load_config()

        # 1. Initialize data manager
        self.data = ProjectData(self.config.project)

        # 2. Initialize the main synthesis pipeline
        self.pipeline = SynthesisPipeline(self.config, self.data)

    def _load_config(self) -> MainConfig:
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return MainConfig(**config_data)

    def run(self) -> List[np.ndarray]:
        """
        Executes the synthesis pipeline and saves the output frames.
        This is the main entry point for the command-line run.py script.

        Returns:
            List[np.ndarray]: The final stylized frames.
        """
        print("\n--- Starting Synthesis Pipeline ---")
        # The pipeline now returns the frames instead of saving them.
        final_frames = self.pipeline.run()

        # The project is responsible for telling the data manager to save the frames.
        self.data.save_output_frames(final_frames)

        print("\n--- Project Execution Complete ---")
        print(f"Output saved to: {self.data.output_dir}")

        return final_frames
