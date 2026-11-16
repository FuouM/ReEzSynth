import logging

import numpy as np

from .config import MainConfig
from .data import ProjectData

# Engine modules import eagerly; their heavy optional dependencies (phycv,
# torchvision, scipy, taichi) load inside the functions that use them.
from .engines.synthesis_engine import EbsynthEngine
from .precompute_runner import PrecomputeRunner
from .pseudo_endpoint import PseudoEndpointGenerator
from .scheduler import SynthesisScheduler

logger = logging.getLogger(__name__)


class SynthesisPipeline:
    """
    Orchestrates the entire video synthesis process. Manages a memory-safe,
    sequential pre-computation workflow with caching, and then delegates to the
    core synthesis engine for the main processing loop.
    """

    def __init__(self, configs: MainConfig, data: ProjectData):
        self.configs = configs
        self.data = data
        # The synthesis engine is used repeatedly, so we initialize it here.
        self.synthesis_engine = EbsynthEngine(
            ebsynth_config=configs.ebsynth_params, pipeline_config=configs.pipeline
        )

    def run(self) -> list[np.ndarray]:
        """
        Main entry point for the synthesis pipeline.
        """
        cfg = self.configs
        logger.info("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()
        use_pseudo_endpoint_styles = cfg.pipeline.use_pseudo_endpoint_styles

        precompute_state = PrecomputeRunner(
            project_cfg=cfg.project,
            precomputation_cfg=cfg.precomputation,
            pipeline_cfg=cfg.pipeline,
            debug_cfg=cfg.debug,
            backend=cfg.ebsynth_params.backend,
        ).run(content_frames)
        style_indices = list(cfg.project.style_indices)
        if use_pseudo_endpoint_styles:
            style_frames, style_indices = PseudoEndpointGenerator(
                engine=self.synthesis_engine,
                precompute_state=precompute_state,
                debug_cfg=cfg.debug,
            ).add_endpoint_styles(
                content_frames=content_frames,
                style_frames=style_frames,
                style_indices=style_indices,
            )

        logger.info("\n--- Starting Synthesis ---")
        final_frames = SynthesisScheduler(
            engine=self.synthesis_engine,
            precompute_state=precompute_state,
            blending_cfg=cfg.blending,
            debug_cfg=cfg.debug,
        ).run(
            content_frames=content_frames,
            style_frames=style_frames,
            style_indices=style_indices,
            force_directional=(
                use_pseudo_endpoint_styles
                and not cfg.pipeline.pseudo_endpoint_use_blending
            ),
        )

        logger.info("\nSynthesis pipeline finished.")
        return final_frames
