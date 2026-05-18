"""Config loading helpers shared by CLI and programmatic entrypoints."""

from pathlib import Path
from typing import Any, Mapping, Union

import yaml

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .service import SynthesisConfigs


def configs_from_mapping(data: Mapping[str, Any]) -> SynthesisConfigs:
    """Build typed config sections from a parsed project config mapping."""
    return SynthesisConfigs(
        project=ProjectConfig(**data["project"]),
        precomputation=PrecomputationConfig(**data["precomputation"]),
        pipeline=PipelineConfig(**data["pipeline"]),
        blending=BlendingConfig(**(data.get("blending") or {})),
        ebsynth_params=EbsynthParamsConfig(**data["ebsynth_params"]),
        debug=DebugConfig(**(data.get("debug") or {})),
    )


def configs_from_yaml(path: Union[str, Path]) -> SynthesisConfigs:
    """Load a YAML project config into typed config sections."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return configs_from_mapping(data)
