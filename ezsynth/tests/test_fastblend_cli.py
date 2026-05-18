import numpy as np

from ezsynth.utils.fastblend_cli import (
    create_fastblend_config,
    create_interpolation_config,
    interpolate_frames,
)
from ezsynth.utils.frame_numbers import (
    extract_frame_number_from_filename,
    match_keyframes_to_frames,
)


def test_fastblend_configs_use_cli_defaults():
    post_config = create_fastblend_config()
    interp_config = create_interpolation_config({"backend": "taichi"})

    assert post_config.enabled is True
    assert post_config.accuracy == 2
    assert post_config.window_size == 5
    assert post_config.batch_size == 16
    assert post_config.minimum_patch_size == 5
    assert post_config.backend == "auto"

    assert interp_config.batch_size == 8
    assert interp_config.minimum_patch_size == 15
    assert interp_config.backend == "taichi"


def test_match_keyframes_to_frames_by_frame_number():
    matched, indices = match_keyframes_to_frames(
        ["frame_00000.png", "frame_00001.png", "frame_00002.png"],
        ["style_00000.png", "style_00002.png"],
    )

    assert extract_frame_number_from_filename("render_00123.jpeg") == 123
    assert matched == ["style_00000.png", None, "style_00002.png"]
    assert indices == [0, 2]


def test_interpolate_frames_handles_single_and_full_coverage_without_engine():
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 255, dtype=np.uint8)
    config = create_interpolation_config()

    single = interpolate_frames([frame_a, frame_b], [frame_b], [1], config)
    full = interpolate_frames([frame_a, frame_b], [frame_a, frame_b], [0, 1], config)

    assert single == [frame_b, frame_b]
    assert full == [frame_a, frame_b]
