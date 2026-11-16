# ezsynth/project.py
import logging

import numpy as np

from .config_io import configs_from_yaml
from .data import ProjectData
from .pipeline import SynthesisPipeline

logger = logging.getLogger(__name__)


class Project:
    def __init__(
        self,
        config_path: str,
        backend: str | None = None,
        forward_warp: bool | None = None,
    ):
        self.configs = configs_from_yaml(config_path)

        # Override backend if specified via command line
        if backend:
            self.configs.ebsynth_params.backend = backend

        # Override forward warp if specified via command line
        if forward_warp is not None:
            self.configs.blending.use_forward_warping = forward_warp
            self.configs.pipeline.use_forward_warping = forward_warp

        self.data = ProjectData(self.configs.project)
        self.pipeline = SynthesisPipeline(self.configs, self.data)

    def run(self) -> list[np.ndarray]:
        """
        Executes the synthesis pipeline and saves the output frames.
        This is the main entry point for the ``reezsynth-video`` CLI.

        Returns:
            List[np.ndarray]: The final stylized frames.
        """
        logger.info("\n--- Starting Synthesis Pipeline ---")
        final_frames = self.pipeline.run()
        self.data.save_output_frames(final_frames)

        logger.info("\n--- Project Execution Complete ---")
        logger.info(f"Output saved to: {self.data.output_dir}")

        return final_frames
