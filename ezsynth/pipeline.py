from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .data import ProjectData

# Engines are imported just-in-time to save memory
from .engines.synthesis_engine import EbsynthEngine
from .precompute import PrecomputeState
from .precompute_runner import PrecomputeRunner
from .utils.blend_utils import Blender
from .utils.sequence_utils import SynthesisSequence, create_sequences
from .utils.warp_utils import PositionalGuide, Warp


class SynthesisPipeline:
    """
    Orchestrates the entire video synthesis process. Manages a memory-safe,
    sequential pre-computation workflow with caching, and then delegates to the
    core synthesis engine for the main processing loop.
    """

    def __init__(
        self,
        ebsynth_params_cfg: EbsynthParamsConfig,
        pipeline_cfg: PipelineConfig,
        project_cfg: ProjectConfig,
        precomputation_cfg: PrecomputationConfig,
        blending_cfg: BlendingConfig,
        data: ProjectData,
        debug_cfg: DebugConfig = None,
    ):
        self.data = data
        self.project_cfg = project_cfg
        self.precomputation_cfg = precomputation_cfg
        self.pipeline_cfg = pipeline_cfg
        self.blending_cfg = blending_cfg
        self.ebsynth_params_cfg = ebsynth_params_cfg
        self.debug_cfg = debug_cfg if debug_cfg is not None else DebugConfig()

        # The synthesis engine is used repeatedly, so we initialize it here.
        # It's a C++ extension and manages its own memory efficiently.
        self.synthesis_engine = EbsynthEngine(
            ebsynth_config=ebsynth_params_cfg, pipeline_config=pipeline_cfg
        )

        self.precompute_state = PrecomputeState()

    def _precompute(self, content_frames: List[np.ndarray]) -> None:
        self.precompute_state = PrecomputeRunner(
            project_cfg=self.project_cfg,
            precomputation_cfg=self.precomputation_cfg,
            pipeline_cfg=self.pipeline_cfg,
            debug_cfg=self.debug_cfg,
            state=self.precompute_state,
        ).run(content_frames)

    def run(self) -> List[np.ndarray]:
        """
        Main entry point for the synthesis pipeline.
        """
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()  # Ensure styles are loaded and resized if needed

        self._precompute(content_frames)

        print("\n--- Starting Synthesis ---")
        final_frames = self._run_synthesis(
            content_frames,
            style_frames,
        )

        print("\nSynthesis pipeline finished.")
        return final_frames

    def _run_synthesis(
        self,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Executes the complete synthesis process (fwd, rev, blend) for the video.
        """
        style_indices = self.project_cfg.style_indices
        sequences = create_sequences(
            num_frames=len(content_frames), style_indices=style_indices
        )

        final_stylized_frames = []
        for seq_idx, seq in enumerate(sequences):
            is_not_first_sequence = seq_idx > 0

            pass_runner_args = {
                "seq": seq,
                "content_frames": content_frames,
            }

            if seq.mode == SynthesisSequence.MODE_FWD:
                style_img = style_frames[seq.style_indices[0]]
                styled_sequence, _, _, _ = self._run_a_pass(
                    **pass_runner_args, style_img=style_img, is_forward=True
                )
                if is_not_first_sequence:
                    styled_sequence.pop(0)
                final_stylized_frames.extend(styled_sequence)

            elif seq.mode == SynthesisSequence.MODE_REV:
                style_img = style_frames[seq.style_indices[0]]
                styled_sequence, _, _, _ = self._run_a_pass(
                    **pass_runner_args, style_img=style_img, is_forward=False
                )
                if is_not_first_sequence:
                    styled_sequence.pop(0)
                final_stylized_frames.extend(styled_sequence)

            elif seq.mode == SynthesisSequence.MODE_BLN:
                style_fwd_idx, style_bwd_idx = (
                    seq.style_indices[0],
                    seq.style_indices[1],
                )
                style_fwd = style_frames[style_fwd_idx]
                style_bwd = style_frames[style_bwd_idx]

                fwd_frames, fwd_err, fwd_flows_used, _ = self._run_a_pass(
                    **pass_runner_args, style_img=style_fwd, is_forward=True
                )
                bwd_frames, bwd_err, _, _ = self._run_a_pass(
                    **pass_runner_args, style_img=style_bwd, is_forward=False
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
                final_sequence = blended_frames + [bwd_frames[-1]]

                if is_not_first_sequence:
                    final_sequence.pop(0)
                final_stylized_frames.extend(final_sequence)

        return final_stylized_frames

    def _prepare_guides_for_frame(
        self,
        keyframe_idx: int,
        target_idx: int,
        style_img: np.ndarray,
        warped_previous_style: np.ndarray,
        source_pos_guide: np.ndarray,
        target_pos_guide: np.ndarray,
        content_frames: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Prepares the list of guide tuples for a single Ebsynth run."""
        eb_params = self.ebsynth_params_cfg
        guides = [
            (
                self.precompute_state.edge_maps[keyframe_idx],
                self.precompute_state.edge_maps[target_idx],
                eb_params.edge_weight,
            ),
            (
                content_frames[keyframe_idx],
                content_frames[target_idx],
                eb_params.image_weight,
            ),
            (source_pos_guide, target_pos_guide, eb_params.pos_weight),
            (style_img, warped_previous_style, eb_params.warp_weight),
        ]

        if self.pipeline_cfg.use_sparse_feature_guide and self.precompute_state.sparse_guides:
            guides.append(
                (
                    self.precompute_state.sparse_guides[keyframe_idx],
                    self.precompute_state.sparse_guides[target_idx],
                    eb_params.sparse_anchor_weight,
                )
            )
        return guides

    def _run_a_pass(
        self,
        seq: SynthesisSequence,
        style_img: np.ndarray,
        is_forward: bool,
        content_frames: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Executes a single forward or reverse synthesis pass for a given sequence.
        """
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
        flows_used_in_pass = []
        nnf_maps = []

        h, w, _ = content_frames[0].shape
        warp = Warp(h, w)
        pos_guider = PositionalGuide(h, w)
        source_pos_guide = pos_guider.get_pristine_guide_uint8()

        previous_nnf = None
        use_propagation = self.pipeline_cfg.use_temporal_nnf_propagation

        for source_idx in tqdm(frame_indices, desc=desc):
            target_idx = source_idx + step

            if is_forward:
                flow = self.precompute_state.fwd_flows[source_idx]
            else:
                flow = self.precompute_state.fwd_flows[target_idx]

            flows_used_in_pass.append(flow)
            previous_stylized_frame = stylized_frames[-1]
            warped_previous_style = warp.run_warping(
                previous_stylized_frame, flow * (-step)
            )
            current_target_pos_guide = PositionalGuide(h, w).create_from_flow(flow)

            guides = self._prepare_guides_for_frame(
                keyframe_idx=keyframe_idx,
                target_idx=target_idx,
                style_img=style_img,
                warped_previous_style=warped_previous_style,
                source_pos_guide=source_pos_guide,
                target_pos_guide=current_target_pos_guide,
                content_frames=content_frames,
            )

            initial_nnf_for_target = None
            if use_propagation and previous_nnf is not None:
                warped_nnf_float = warp.run_warping(
                    previous_nnf.astype(np.float32), flow * (-step)
                )
                initial_nnf_for_target = warped_nnf_float.astype(np.int32)

            run_output = self.synthesis_engine.run(
                style_img,
                guides=guides,
                initial_nnf=initial_nnf_for_target,
                output_nnf=use_propagation,
            )

            if use_propagation:
                stylized_img, err, nnf = run_output
                previous_nnf = nnf
                nnf_maps.append(nnf)
            else:
                stylized_img, err = run_output

            stylized_frames.append(stylized_img)
            error_maps.append(err)

        if not is_forward:
            stylized_frames.reverse()
            error_maps.reverse()
            flows_used_in_pass.reverse()
            nnf_maps.reverse()

        return stylized_frames, error_maps, flows_used_in_pass, nnf_maps
