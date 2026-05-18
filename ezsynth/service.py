"""Composable synthesis service types shared by CLI/API/future integrations."""

from dataclasses import dataclass

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)


@dataclass
class SynthesisConfigs:
    """All config sections needed to construct a synthesis pipeline."""

    project: ProjectConfig
    precomputation: PrecomputationConfig
    pipeline: PipelineConfig
    blending: BlendingConfig
    ebsynth_params: EbsynthParamsConfig
    debug: DebugConfig
