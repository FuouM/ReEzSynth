import numpy as np

from ezsynth.config import DebugConfig, EbsynthParamsConfig, PipelineConfig
from ezsynth.pseudo_endpoint import PseudoEndpointGenerator
from ezsynth.precompute import PrecomputeState
from ezsynth.utils.occlusion import (
    accumulate_target_to_source_coords,
    build_dfs_pseudo_style,
)


class _FakeEngine:
    def __init__(self, pipeline_config=None):
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.ebsynth_config = EbsynthParamsConfig(backend="torch")


def test_accumulate_target_to_source_coords_uses_forward_flow():
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[..., 0] = 1.0

    coords, valid = accumulate_target_to_source_coords(
        height=2,
        width=2,
        target_idx=0,
        source_idx=1,
        fwd_flows=[flow],
        bwd_flows=[np.zeros_like(flow)],
    )

    assert coords[0, 0, 0] == 1.0
    assert not valid[0, 1]


def test_build_dfs_pseudo_style_assigns_confident_region():
    style = np.full((3, 3, 3), 200, dtype=np.uint8)
    content = np.zeros((3, 3, 3), dtype=np.uint8)
    y, x = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    coords = np.stack((x, y), axis=-1).astype(np.float32)

    pseudo, confidence = build_dfs_pseudo_style(
        style_img=style,
        source_content=content,
        target_content=content.copy(),
        source_coords=coords,
        valid_coords=np.ones((3, 3), dtype=bool),
        min_region_size=1,
    )

    np.testing.assert_array_equal(pseudo, style)
    assert np.all(confidence == 255)


def test_pseudo_endpoint_generator_adds_missing_endpoints():
    content = [np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(3)]
    style = np.full((3, 3, 3), 128, dtype=np.uint8)
    zero_flow = np.zeros((3, 3, 2), dtype=np.float32)
    generator = PseudoEndpointGenerator(
        engine=_FakeEngine(
            PipelineConfig(
                use_pseudo_endpoint_styles=True,
                pseudo_endpoint_dfs_min_region_size=1,
            )
        ),
        precompute_state=PrecomputeState(
            fwd_flows=[zero_flow, zero_flow.copy()],
            bwd_flows=[zero_flow.copy(), zero_flow.copy()],
        ),
        debug_cfg=DebugConfig(),
    )

    frames, indices = generator.add_endpoint_styles(
        content_frames=content,
        style_frames=[style],
        style_indices=[1],
    )

    assert indices == [0, 1, 2]
    assert len(frames) == 3
    np.testing.assert_array_equal(frames[1], style)
