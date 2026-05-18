"""Video synthesis scheduling and blend orchestration."""

from typing import List

import numpy as np

from .config import BlendingConfig, DebugConfig
from .engines.pass_runner import SynthesisPassRunner
from .engines.synthesis_engine import EbsynthEngine
from .precompute import PrecomputeState
from .utils.blend_utils import Blender
from .utils.sequence_utils import (
    SynthesisSequence,
    create_directional_sequences,
    create_sequences,
)


class SynthesisScheduler:
    """Run scheduled synthesis sequences and blend bidirectional intervals."""

    def __init__(
        self,
        *,
        engine: EbsynthEngine,
        precompute_state: PrecomputeState,
        blending_cfg: BlendingConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        self.engine = engine
        self.precompute_state = precompute_state
        self.blending_cfg = blending_cfg
        self.debug_cfg = debug_cfg
        self.pass_runner = SynthesisPassRunner(
            engine=self.engine,
            precompute_state=self.precompute_state,
            debug_cfg=self.debug_cfg,
        )

    def run(
        self,
        *,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
        style_indices: List[int],
        force_directional: bool = False,
    ) -> List[np.ndarray]:
        sequence_builder = create_directional_sequences if force_directional else create_sequences
        sequences = sequence_builder(
            num_frames=len(content_frames),
            style_indices=style_indices,
        )

        final_stylized_frames = []
        for seq_idx, seq in enumerate(sequences):
            final_sequence = self._run_sequence(
                seq=seq,
                content_frames=content_frames,
                style_frames=style_frames,
            )
            if seq_idx > 0:
                final_sequence.pop(0)
            final_stylized_frames.extend(final_sequence)

        return final_stylized_frames

    def _run_sequence(
        self,
        *,
        seq: SynthesisSequence,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        pass_runner_args = {
            "seq": seq,
            "content_frames": content_frames,
        }

        if seq.mode in (SynthesisSequence.MODE_FWD, SynthesisSequence.MODE_REV):
            is_forward = seq.mode == SynthesisSequence.MODE_FWD
            style_img = style_frames[seq.style_indices[0]]
            styled_sequence, _, _, _ = self.pass_runner.run(
                **pass_runner_args,
                style_img=style_img,
                is_forward=is_forward,
                collect_intermediates=False,
            )
            return styled_sequence

        if seq.mode == SynthesisSequence.MODE_BLN:
            return self._run_blend_sequence(
                seq=seq,
                content_frames=content_frames,
                style_frames=style_frames,
            )

        raise ValueError(f"Unsupported synthesis sequence mode: {seq.mode!r}")

    def _run_blend_sequence(
        self,
        *,
        seq: SynthesisSequence,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        style_fwd_idx, style_bwd_idx = seq.style_indices[0], seq.style_indices[1]
        fwd_frames, fwd_err, fwd_flows_used, _ = self.pass_runner.run(
            seq=seq,
            content_frames=content_frames,
            style_img=style_frames[style_fwd_idx],
            is_forward=True,
        )
        bwd_frames, bwd_err, _, _ = self.pass_runner.run(
            seq=seq,
            content_frames=content_frames,
            style_img=style_frames[style_bwd_idx],
            is_forward=False,
        )

        h, w, _ = content_frames[0].shape
        blender = Blender(h, w, **self.blending_cfg.model_dump())
        blended_frames = blender.run(
            fwd_frames=fwd_frames,
            bwd_frames=bwd_frames,
            fwd_errors=fwd_err,
            bwd_errors=bwd_err,
            fwd_flows=fwd_flows_used,
        )
        return blended_frames + [bwd_frames[-1]]
