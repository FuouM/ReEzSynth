from typing import List

import numpy as np

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .data import ProjectData

# Engines are imported just-in-time to save memory
from .engines.synthesis_engine import EbsynthEngine
from .precompute import PrecomputeState
from .precompute_runner import PrecomputeRunner
from .pseudo_endpoint import PseudoEndpointGenerator
from .scheduler import SynthesisScheduler


class SynthesisPipeline:
    """
    Orchestrates the entire video synthesis process. Manages a memory-safe,
    sequential pre-computation workflow with caching, and then delegates to the
    core synthesis engine for the main processing loop.
    """

    def __init__(
        self,
        ebsynth_params_cfg: EbsynthParamsConfig,
        pipeline_cfg: PipelineConfig,
        project_cfg: ProjectConfig,
        precomputation_cfg: PrecomputationConfig,
        blending_cfg: BlendingConfig,
        data: ProjectData,
        debug_cfg: DebugConfig = None,
    ):
        self.data = data
        self.project_cfg = project_cfg
        self.precomputation_cfg = precomputation_cfg
        self.pipeline_cfg = pipeline_cfg
        self.blending_cfg = blending_cfg
        self.ebsynth_params_cfg = ebsynth_params_cfg
        self.debug_cfg = debug_cfg if debug_cfg is not None else DebugConfig()

        # The synthesis engine is used repeatedly, so we initialize it here.
        # It's a C++ extension and manages its own memory efficiently.
        self.synthesis_engine = EbsynthEngine(
            ebsynth_config=ebsynth_params_cfg, pipeline_config=pipeline_cfg
        )

        self.precompute_state = PrecomputeState()

    def _precompute(self, content_frames: List[np.ndarray]) -> None:
        self.precompute_state = PrecomputeRunner(
            project_cfg=self.project_cfg,
            precomputation_cfg=self.precomputation_cfg,
            pipeline_cfg=self.pipeline_cfg,
            debug_cfg=self.debug_cfg,
            state=self.precompute_state,
        ).run(content_frames)

    def run(self) -> List[np.ndarray]:
        """
        Main entry point for the synthesis pipeline.
        """
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()  # Ensure styles are loaded and resized if needed
        use_pseudo_endpoint_styles = self.pipeline_cfg.use_pseudo_endpoint_styles

        self._precompute(content_frames)
        style_indices = list(self.project_cfg.style_indices)
        if use_pseudo_endpoint_styles:
            style_frames, style_indices = PseudoEndpointGenerator(
                engine=self.synthesis_engine,
                precompute_state=self.precompute_state,
                debug_cfg=self.debug_cfg,
            ).add_endpoint_styles(
                content_frames=content_frames,
                style_frames=style_frames,
                style_indices=style_indices,
            )

        print("\n--- Starting Synthesis ---")
        final_frames = SynthesisScheduler(
            engine=self.synthesis_engine,
            precompute_state=self.precompute_state,
            blending_cfg=self.blending_cfg,
            debug_cfg=self.debug_cfg,
        ).run(
            content_frames=content_frames,
            style_frames=style_frames,
            style_indices=style_indices,
            force_directional=(
                use_pseudo_endpoint_styles
                and not self.pipeline_cfg.pseudo_endpoint_use_blending
            ),
        )

        print("\nSynthesis pipeline finished.")
        return final_frames
