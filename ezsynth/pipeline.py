# ezsynth/pipeline.py
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .config import MainConfig
from .data import ProjectData
from .engines.edge_engine import EdgeEngine
from .engines.flow_engine import RAFTFlowEngine
from .engines.synthesis_engine import EbsynthEngine
from .utils.blend_utils import Blender
from .utils.sequence_utils import SynthesisSequence, create_sequences
from .utils.warp_utils import PositionalGuide, Warp


class SynthesisPipeline:
    def __init__(self, config: MainConfig, data: ProjectData):
        self.config = config
        self.data = data

        # Engines are now owned by the pipeline
        self.synthesis_engine = EbsynthEngine(config.ebsynth_params)
        self.flow_engine = RAFTFlowEngine(
            model_name=config.precomputation.flow_model,
            arch=config.precomputation.flow_engine,
        )
        self.edge_engine = EdgeEngine(method=config.precomputation.edge_method)

        # In-memory caches for computed data
        self._edge_maps = None
        self._fwd_flows = None
        self._rev_flows = None

    def _compute_all_data(self, content_frames: List[np.ndarray]):
        """Computes all necessary data upfront and stores it in memory."""
        print("Computing edge maps...")
        self._edge_maps = self.edge_engine.compute(content_frames)

        print("Computing forward optical flow...")
        self._fwd_flows = self.flow_engine.compute(content_frames)

        print("Computing reverse optical flow...")
        self._rev_flows = self.flow_engine.compute_reverse(content_frames)
        print("All pre-computation finished.")

    def run(self):
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()
        self._compute_all_data(content_frames)
        self.content_frames = content_frames
        self.style_frames = style_frames

        style_indices = self.config.project.style_indices
        sequences = create_sequences(
            num_frames=len(content_frames), style_indices=style_indices
        )

        final_stylized_frames = []
        for seq_idx, seq in enumerate(sequences):
            is_not_first_sequence = seq_idx > 0

            if seq.mode == SynthesisSequence.MODE_FWD:
                style_img = self.style_frames[seq.style_indices[0]]
                styled_sequence, _, _ = self._run_a_pass(
                    seq, style_img, is_forward=True
                )
                if is_not_first_sequence:
                    styled_sequence.pop(0)
                final_stylized_frames.extend(styled_sequence)

            elif seq.mode == SynthesisSequence.MODE_REV:
                style_img = self.style_frames[seq.style_indices[0]]
                styled_sequence, _, _ = self._run_a_pass(
                    seq, style_img, is_forward=False
                )
                if is_not_first_sequence:
                    styled_sequence.pop(0)
                final_stylized_frames.extend(styled_sequence)

            elif seq.mode == SynthesisSequence.MODE_BLN:
                print(f"\nProcessing blend sequence: {seq}")
                style_fwd_idx, style_bwd_idx = (
                    seq.style_indices[0],
                    seq.style_indices[1],
                )
                style_fwd = self.style_frames[style_fwd_idx]
                style_bwd = self.style_frames[style_bwd_idx]

                fwd_frames, fwd_err, fwd_flows = self._run_a_pass(
                    seq, style_fwd, is_forward=True
                )
                bwd_frames, bwd_err, _ = self._run_a_pass(
                    seq, style_bwd, is_forward=False
                )

                h, w, _ = self.content_frames[0].shape
                blender = Blender(h, w, use_lsqr=True)

                # Blender now returns N-1 frames, where the first one is the blended keyframe.
                blended_frames = blender.run(
                    fwd_frames=fwd_frames,
                    bwd_frames=bwd_frames,
                    fwd_errors=fwd_err,
                    bwd_errors=bwd_err,
                    fwd_flows=fwd_flows,
                )

                # --- FINAL ASSEMBLY FIX ---
                # Replicate the old code's assembly:
                # The final sequence is the N-1 blended frames + the pristine end keyframe.
                # This correctly replaces the start keyframe with its blended version.
                final_sequence = blended_frames + [bwd_frames[-1]]

                if is_not_first_sequence:
                    final_sequence.pop(0)
                final_stylized_frames.extend(final_sequence)

        self.data.save_output_frames(final_stylized_frames)
        print("\nSynthesis pipeline with blending finished.")

    def _run_a_pass(
        self, seq: SynthesisSequence, style_img: np.ndarray, is_forward: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if is_forward:
            frame_indices = range(seq.start_frame, seq.end_frame)
            step = 1
            desc = f"Forward Pass (Frames {seq.start_frame}-{seq.end_frame})"
            keyframe_idx = seq.start_frame
        else:
            frame_indices = range(seq.end_frame, seq.start_frame, -1)
            step = -1
            desc = f"Reverse Pass (Frames {seq.end_frame}-{seq.start_frame})"
            keyframe_idx = seq.end_frame

        stylized_frames = [style_img]
        error_maps = []
        flows_used = []

        h, w, _ = self.content_frames[0].shape
        warp = Warp(h, w)

        pos_guider = PositionalGuide(h, w)
        source_pos_guide = None

        for i in tqdm(frame_indices, desc=desc):
            if is_forward:
                source_idx, target_idx = i, i + 1
                flow = self._fwd_flows[source_idx]
            else:
                source_idx, target_idx = i, i - 1
                flow = self._fwd_flows[target_idx]

            flows_used.append(flow)
            previous_stylized_frame = stylized_frames[-1]
            warped_previous_style = warp.run_warping(
                previous_stylized_frame, flow * (-step)
            )
            current_target_pos_guide = pos_guider.create_from_flow(flow)
            if source_pos_guide is None:
                source_pos_guide = pos_guider.get_pristine_guide_uint8()

            target_content_frame = self.content_frames[target_idx]
            target_edge_map = self._edge_maps[target_idx]
            eb_params = self.config.ebsynth_params

            guides_for_ebsynth = [
                (self._edge_maps[keyframe_idx], target_edge_map, eb_params.edge_weight),
                (
                    self.content_frames[keyframe_idx],
                    target_content_frame,
                    eb_params.image_weight,
                ),
                (source_pos_guide, current_target_pos_guide, eb_params.pos_weight),
                (style_img, warped_previous_style, eb_params.warp_weight),
            ]

            stylized_img, err = self.synthesis_engine.eb.run(
                style_img, guides=guides_for_ebsynth
            )

            stylized_frames.append(stylized_img)
            error_maps.append(err)

        if not is_forward:
            stylized_frames.reverse()
            error_maps.reverse()
            flows_used.reverse()

        return stylized_frames, error_maps, flows_used
