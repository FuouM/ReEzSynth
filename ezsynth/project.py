# ezsynth/project.py
from pathlib import Path
from typing import List

import numpy as np

from .config_io import configs_from_yaml
from .data import ProjectData
from .output import OutputManager
from .pipeline import SynthesisPipeline
from .service import SynthesisConfigs


class Project:
    def __init__(self, config_path: str, backend: str = None):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        self.configs = self._load_config()

        # Override backend if specified via command line
        if backend:
            self.configs.ebsynth_params.backend = backend

        # 1. Initialize data manager
        self.data = ProjectData.from_config(self.configs.project)

        # 2. Initialize the main synthesis pipeline
        self.pipeline = SynthesisPipeline(
            ebsynth_params_cfg=self.configs.ebsynth_params,
            pipeline_cfg=self.configs.pipeline,
            project_cfg=self.configs.project,
            precomputation_cfg=self.configs.precomputation,
            blending_cfg=self.configs.blending,
            data=self.data,
            debug_cfg=self.configs.debug,
        )

    def _load_config(self) -> SynthesisConfigs:
        return configs_from_yaml(self.config_path)

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

        OutputManager(self.data).handle(
            final_frames,
            save=True,
            visible_output_dir=self.data.output_dir,
        )

        print("\n--- Project Execution Complete ---")
        print(f"Output saved to: {self.data.output_dir}")

        return final_frames
