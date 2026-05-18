"""Composable synthesis service types shared by CLI/API/future integrations."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

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
from .output import OutputManager
from .pipeline import SynthesisPipeline


@dataclass
class SynthesisConfigs:
    """All config sections needed to construct a synthesis pipeline."""

    project: ProjectConfig
    precomputation: PrecomputationConfig
    pipeline: PipelineConfig
    blending: BlendingConfig
    ebsynth_params: EbsynthParamsConfig
    debug: DebugConfig


@dataclass
class SynthesisRequest:
    """Integration-friendly request object for path-based synthesis."""

    content_dir: str
    style_paths: Sequence[str | Path]
    style_indices: Sequence[int]
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    mask_dir: Optional[str] = None
    modulation_dir: Optional[str] = None
    project_name: Optional[str] = None
    save_outputs: bool = True
    ebsynth_params: Optional[EbsynthParamsConfig] = None
    pipeline: Optional[PipelineConfig] = None
    blending: Optional[BlendingConfig] = None
    precomputation: Optional[PrecomputationConfig] = None
    debug: Optional[DebugConfig] = None

    def build_configs(self, *, output_dir: str, cache_dir: str) -> SynthesisConfigs:
        project = ProjectConfig(
            name=self.project_name or Path(self.content_dir).name,
            content_dir=self.content_dir,
            style_path=[str(path) for path in self.style_paths],
            style_indices=list(self.style_indices),
            output_dir=output_dir,
            cache_dir=cache_dir,
            mask_dir=self.mask_dir,
            modulation_dir=self.modulation_dir,
        )
        return SynthesisConfigs(
            project=project,
            precomputation=self.precomputation or PrecomputationConfig(),
            pipeline=self.pipeline or PipelineConfig(),
            blending=self.blending or BlendingConfig(),
            ebsynth_params=self.ebsynth_params or EbsynthParamsConfig(),
            debug=self.debug or DebugConfig(),
        )


@dataclass
class SynthesisResult:
    """Frames and resolved paths produced by a synthesis run."""

    frames: List[np.ndarray]
    output_dir: Optional[Path]
    cache_dir: Optional[Path]
    saved: bool


class SynthesisService:
    """Small orchestration layer reusable by CLI, API, and future integrations."""

    def run(self, request: SynthesisRequest) -> SynthesisResult:
        with _temp_dir_if_missing(request.output_dir) as output_dir, _temp_dir_if_missing(
            request.cache_dir
        ) as cache_dir:
            configs = request.build_configs(output_dir=output_dir, cache_dir=cache_dir)
            data = ProjectData.from_config(configs.project)
            pipeline = self.build_pipeline(configs, data)
            frames = pipeline.run()

            should_save = request.save_outputs and request.output_dir is not None
            output = OutputManager(data).handle(
                frames,
                save=should_save,
                visible_output_dir=Path(output_dir)
                if request.output_dir is not None
                else None,
            )

            return SynthesisResult(
                frames=output.frames,
                output_dir=output.output_dir,
                cache_dir=Path(cache_dir) if request.cache_dir is not None else None,
                saved=output.saved,
            )

    def build_pipeline(
        self, configs: SynthesisConfigs, data: ProjectData
    ) -> SynthesisPipeline:
        return SynthesisPipeline(
            ebsynth_params_cfg=configs.ebsynth_params,
            pipeline_cfg=configs.pipeline,
            project_cfg=configs.project,
            precomputation_cfg=configs.precomputation,
            blending_cfg=configs.blending,
            data=data,
            debug_cfg=configs.debug,
        )


class _temp_dir_if_missing:
    def __init__(self, path: Optional[str]) -> None:
        self._path = path
        self._temp: Optional[tempfile.TemporaryDirectory] = None

    def __enter__(self) -> str:
        if self._path is not None:
            return self._path
        self._temp = tempfile.TemporaryDirectory()
        return self._temp.name

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._temp is not None:
            self._temp.cleanup()
