from pathlib import Path

import numpy as np

from ezsynth.integration_io import (
    bgr_uint8_to_rgb_float,
    rgb_float_to_bgr_uint8,
    write_rgb_frame_sequence,
)
from ezsynth.output import OutputManager
from ezsynth.service import SynthesisRequest, SynthesisService


class _FakePipeline:
    def run(self):
        return [np.zeros((2, 2, 3), dtype=np.uint8)]


def test_request_builds_typed_configs_from_integration_paths(tmp_path: Path):
    request = SynthesisRequest(
        content_dir=str(tmp_path / "content"),
        style_paths=[tmp_path / "style.png"],
        style_indices=[0],
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
    )

    configs = request.build_configs(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
    )

    assert configs.project.content_dir == str(tmp_path / "content")
    assert configs.project.style_path == [str(tmp_path / "style.png")]
    assert configs.project.output_dir == str(tmp_path / "out")


def test_service_run_can_skip_saving_for_in_memory_consumers(
    tmp_path: Path, monkeypatch
):
    request = SynthesisRequest(
        content_dir=str(tmp_path / "content"),
        style_paths=[str(tmp_path / "style.png")],
        style_indices=[0],
        cache_dir=str(tmp_path / "cache"),
        save_outputs=False,
    )
    service = SynthesisService()
    monkeypatch.setattr(
        service, "build_pipeline", lambda configs, data: _FakePipeline()
    )

    result = service.run(request)

    assert len(result.frames) == 1
    assert result.output_dir is None
    assert result.cache_dir == tmp_path / "cache"
    assert not result.saved


def test_output_manager_can_return_frames_without_saving(tmp_path: Path):
    class _Data:
        output_dir = tmp_path / "out"

        def save_output_frames(self, frames):
            raise AssertionError("save should not be called")

    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    result = OutputManager(_Data()).handle(
        frames,
        save=False,
        visible_output_dir=tmp_path / "out",
    )

    assert result.frames is frames
    assert result.output_dir is None
    assert not result.saved


def test_integration_io_converts_rgb_float_and_writes_sequence(tmp_path: Path):
    rgb = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)

    bgr = rgb_float_to_bgr_uint8(rgb)
    roundtrip = bgr_uint8_to_rgb_float(bgr)
    paths = write_rgb_frame_sequence([rgb], tmp_path / "frames")

    assert bgr.tolist() == [[[128, 0, 255]]]
    np.testing.assert_allclose(roundtrip, np.array([[[1.0, 0.0, 128 / 255]]]))
    assert paths.directory == tmp_path / "frames"
    assert len(paths.frame_paths) == 1
    assert paths.frame_paths[0].exists()
