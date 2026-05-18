import numpy as np

from ezsynth.config import BlendingConfig, DebugConfig, EbsynthParamsConfig, PipelineConfig
from ezsynth.engines.pass_runner import SynthesisPassRunner
from ezsynth.precompute import PrecomputeState
from ezsynth.scheduler import SynthesisScheduler
from ezsynth.utils.sequence_utils import (
    SynthesisSequence,
    create_directional_sequences,
)
from ezsynth.utils.warp_utils import Warp


class _FakeEngine:
    def __init__(self, pipeline_config=None):
        self.ebsynth_config = EbsynthParamsConfig(backend="torch")
        self.pipeline_config = pipeline_config or PipelineConfig(
            use_temporal_nnf_propagation=False
        )
        self.calls = []

    def run(
        self,
        style_img,
        guides,
        modulation_map=None,
        initial_nnf=None,
        output_nnf=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "style_img": style_img,
                "guides": guides,
                "modulation_map": modulation_map,
                "initial_nnf": initial_nnf,
                "output_nnf": output_nnf,
            }
        )
        target = guides[1].target
        error = np.zeros(target.shape[:2], dtype=np.float32)
        return target.copy(), error


def test_pass_runner_uses_precomputed_guides_and_flow(monkeypatch):
    class _FakeContext:
        def __init__(self, engine, style_img, guides):
            self.engine = engine
            self.style_img = style_img

        def run_frame(
            self,
            guides,
            modulation_map=None,
            initial_nnf=None,
            return_error=True,
            output_nnf=False,
        ):
            return self.engine.run(
                self.style_img,
                guides=guides,
                modulation_map=modulation_map,
                initial_nnf=initial_nnf,
                output_nnf=output_nnf,
            )

    monkeypatch.setattr("ezsynth.engines.pass_runner.PreparedSynthesisContext", _FakeContext)

    style = np.full((2, 2, 3), 255, dtype=np.uint8)
    content = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 10, dtype=np.uint8),
    ]
    state = PrecomputeState(
        edge_maps=[frame.copy() for frame in content],
        fwd_flows=[np.zeros((2, 2, 2), dtype=np.float32)],
    )
    engine = _FakeEngine()
    runner = SynthesisPassRunner(
        engine=engine,
        precompute_state=state,
        debug_cfg=DebugConfig(),
    )

    frames, errors, flows, nnfs = runner.run(
        seq=SynthesisSequence(0, 1, SynthesisSequence.MODE_FWD, [0]),
        style_img=style,
        is_forward=True,
        content_frames=content,
    )

    assert len(frames) == 2
    assert len(errors) == 1
    assert flows[0] is state.fwd_flows[0]
    assert nnfs == []
    assert len(engine.calls[0]["guides"]) == 4
    assert engine.calls[0]["guides"][0].weight == engine.ebsynth_config.edge_weight
    np.testing.assert_array_equal(frames[-1], content[-1])


def test_pass_runner_uses_occlusion_modulation(monkeypatch):
    class _FakeContext:
        def __init__(self, engine, style_img, guides):
            self.engine = engine
            self.style_img = style_img

        def run_frame(
            self,
            guides,
            modulation_map=None,
            initial_nnf=None,
            return_error=True,
            output_nnf=False,
        ):
            return self.engine.run(
                self.style_img,
                guides=guides,
                modulation_map=modulation_map,
                initial_nnf=initial_nnf,
                output_nnf=output_nnf,
            )

    monkeypatch.setattr("ezsynth.engines.pass_runner.PreparedSynthesisContext", _FakeContext)

    style = np.full((2, 2, 3), 255, dtype=np.uint8)
    content = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 10, dtype=np.uint8),
    ]
    mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    state = PrecomputeState(
        edge_maps=[frame.copy() for frame in content],
        fwd_flows=[np.zeros((2, 2, 2), dtype=np.float32)],
        fwd_occlusion_masks=[mask],
    )
    engine = _FakeEngine(
        PipelineConfig(
            use_temporal_nnf_propagation=False,
            use_flow_occlusion_modulation=True,
            occlusion_modulation_floor=32,
        )
    )
    runner = SynthesisPassRunner(
        engine=engine,
        precompute_state=state,
        debug_cfg=DebugConfig(),
    )

    runner.run(
        seq=SynthesisSequence(0, 1, SynthesisSequence.MODE_FWD, [0]),
        style_img=style,
        is_forward=True,
        content_frames=content,
    )

    modulation = engine.calls[0]["modulation_map"]
    assert modulation[0, 0, 0] == 255
    assert modulation[0, 1, 0] == 32


def test_pass_runner_uses_forward_warping_when_enabled(monkeypatch):
    class _FakeContext:
        def __init__(self, engine, style_img, guides):
            self.engine = engine
            self.style_img = style_img

        def run_frame(
            self,
            guides,
            modulation_map=None,
            initial_nnf=None,
            return_error=True,
            output_nnf=False,
        ):
            return self.engine.run(
                self.style_img,
                guides=guides,
                modulation_map=modulation_map,
                initial_nnf=initial_nnf,
                output_nnf=output_nnf,
            )

    calls = []

    def _fake_forward(self, img, flow, **kwargs):
        calls.append(flow.copy())
        return img.copy()

    monkeypatch.setattr("ezsynth.engines.pass_runner.PreparedSynthesisContext", _FakeContext)
    monkeypatch.setattr(Warp, "run_forward_warping", _fake_forward)

    style = np.full((2, 2, 3), 255, dtype=np.uint8)
    content = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 10, dtype=np.uint8),
    ]
    flow = np.ones((2, 2, 2), dtype=np.float32)
    state = PrecomputeState(
        edge_maps=[frame.copy() for frame in content],
        fwd_flows=[flow],
    )
    engine = _FakeEngine(
        PipelineConfig(
            use_temporal_nnf_propagation=False,
            use_forward_warping=True,
        )
    )

    SynthesisPassRunner(
        engine=engine,
        precompute_state=state,
        debug_cfg=DebugConfig(),
    ).run(
        seq=SynthesisSequence(0, 1, SynthesisSequence.MODE_FWD, [0]),
        style_img=style,
        is_forward=True,
        content_frames=content,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], flow)


def test_scheduler_removes_duplicate_sequence_boundaries(monkeypatch):
    class _FakePassRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, *, seq, style_img, is_forward, content_frames, **kwargs):
            frame = np.full((1, 1, 3), seq.start_frame, dtype=np.uint8)
            next_frame = np.full((1, 1, 3), seq.end_frame, dtype=np.uint8)
            return [frame, next_frame], [], [], []

    monkeypatch.setattr("ezsynth.scheduler.SynthesisPassRunner", _FakePassRunner)

    content = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(3)]
    style = [np.zeros((1, 1, 3), dtype=np.uint8)]
    scheduler = SynthesisScheduler(
        engine=_FakeEngine(),
        precompute_state=PrecomputeState(),
        blending_cfg=BlendingConfig(),
        debug_cfg=DebugConfig(),
    )

    frames = scheduler.run(
        content_frames=content,
        style_frames=style,
        style_indices=[0],
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 2]


def test_directional_sequences_replace_blend_intervals():
    sequences = create_directional_sequences(num_frames=4, style_indices=[0, 2, 3])

    assert [seq.mode for seq in sequences] == [
        SynthesisSequence.MODE_FWD,
        SynthesisSequence.MODE_FWD,
    ]
    assert [seq.style_indices for seq in sequences] == [[0], [1]]
