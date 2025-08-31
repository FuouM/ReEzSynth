# ezsynth/project.py
from pathlib import Path

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

    def run(self):
        print("\n--- Starting Synthesis Pipeline ---")
        self.pipeline.run()

        print("\n--- Project Execution Complete ---")
        print(f"Output saved to: {self.data.output_dir}")
