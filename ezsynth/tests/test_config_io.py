from pathlib import Path

import pytest

from ezsynth.config import BlendingConfig, DebugConfig, EbsynthParamsConfig
from ezsynth.config import PrecomputationConfig
from ezsynth.config_io import configs_from_mapping, configs_from_yaml
from ezsynth.service import SynthesisConfigs


def _minimal_config_data():
    return {
        "project": {
            "content_dir": "content",
            "style_path": "style.png",
            "style_indices": [0],
            "output_dir": "output",
        },
        "precomputation": {},
        "pipeline": {},
        "ebsynth_params": {},
    }


def test_configs_from_mapping_builds_config_sections():
    configs = configs_from_mapping(_minimal_config_data())

    assert isinstance(configs, SynthesisConfigs)
    assert configs.project.content_dir == "content"
    assert isinstance(configs.blending, BlendingConfig)
    assert isinstance(configs.ebsynth_params, EbsynthParamsConfig)
    assert isinstance(configs.debug, DebugConfig)
    assert configs.blending.poisson_solver == "lsqr"
    assert configs.precomputation.flow_engine == "NeuFlow"
    assert configs.precomputation.flow_model == "neuflow_mixed"
    assert configs.debug.save_flow_viz is False


def test_configs_from_yaml_loads_config_sections(tmp_path: Path):
    config_path = tmp_path / "project.yml"
    config_path.write_text(
        """
project:
  content_dir: content
  style_path: style.png
  style_indices: [0]
  output_dir: output
precomputation: {}
pipeline: {}
ebsynth_params: {}
""".lstrip(),
        encoding="utf-8",
    )

    configs = configs_from_yaml(config_path)

    assert isinstance(configs, SynthesisConfigs)
    assert configs.project.output_dir == "output"


def test_configs_from_yaml_requires_existing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        configs_from_yaml(tmp_path / "missing.yml")


def test_configs_from_yaml_requires_mapping(tmp_path: Path):
    config_path = tmp_path / "project.yml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        configs_from_yaml(config_path)


def test_precomputation_config_validates_flow_model_matches_engine():
    with pytest.raises(ValueError, match="RAFT checkpoint"):
        PrecomputationConfig(flow_engine="RAFT", flow_model="neuflow_mixed")

    with pytest.raises(ValueError, match="NeuFlow checkpoint"):
        PrecomputationConfig(flow_engine="NeuFlow", flow_model="sintel")


def test_precomputation_config_accepts_alternate_flow_engines():
    opencv = PrecomputationConfig(flow_engine="OpenCV")
    torchvision = PrecomputationConfig(flow_engine="TorchVision")

    assert opencv.opencv_flow_method == "DIS"
    assert torchvision.torchvision_flow_model == "raft_large"
