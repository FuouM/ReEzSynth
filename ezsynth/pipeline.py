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
    """
    Orchestrates the entire video synthesis process by managing data loading,
    pre-computation, and delegating to the core synthesis engine.
    """

    def __init__(self, config: MainConfig, data: ProjectData):
        self.config = config
        self.data = data

        self.synthesis_engine = EbsynthEngine(
            ebsynth_config=config.ebsynth_params, pipeline_config=config.pipeline
        )
        self.flow_engine = RAFTFlowEngine(
            model_name=config.precomputation.flow_model,
            arch=config.precomputation.flow_engine,
        )
        self.edge_engine = EdgeEngine(method=config.precomputation.edge_method)

        self._edge_maps: List[np.ndarray] = []
        self._fwd_flows: List[np.ndarray] = []

    def _compute_all_data(self, content_frames: List[np.ndarray]):
        """Computes all necessary data upfront and stores it in memory."""
        print("Computing edge maps...")
        self._edge_maps = self.edge_engine.compute(content_frames)
        print("Computing forward optical flow (i -> i+1)...")
        self._fwd_flows = self.flow_engine.compute(content_frames)
        print("All pre-computation finished.")

    def run(self):
        """
        Main entry point for the synthesis pipeline.
        """
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()
        modulation_frames = self.data.get_modulation_frames()
        self._compute_all_data(content_frames)

        print("\n--- Starting Synthesis ---")
        final_frames = self._run_synthesis(
            content_frames,
            style_frames,
            self._edge_maps,
            self._fwd_flows,
            modulation_frames,
        )

        self.data.save_output_frames(final_frames)
        print("\nSynthesis pipeline finished.")

    def _run_synthesis(
        self,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
        edge_maps: List[np.ndarray],
        fwd_flows: List[np.ndarray],
        modulation_frames: List[np.ndarray] | None,
    ) -> List[np.ndarray]:
        """
        Executes the complete synthesis process (fwd, rev, blend) for the video.
        """
        style_indices = self.config.project.style_indices
        sequences = create_sequences(
            num_frames=len(content_frames), style_indices=style_indices
        )

        final_stylized_frames = []
        for seq_idx, seq in enumerate(sequences):
            is_not_first_sequence = seq_idx > 0

            pass_runner_args = {
                "seq": seq,
                "content_frames": content_frames,
                "edge_maps": edge_maps,
                "fwd_flows": fwd_flows,
                "modulation_frames": modulation_frames,
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
                blender = Blender(h, w, **self.config.blending.model_dump())

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
        edge_maps: List[np.ndarray],
        content_frames: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Prepares the list of guide tuples for a single Ebsynth run."""
        eb_params = self.config.ebsynth_params
        return [
            (edge_maps[keyframe_idx], edge_maps[target_idx], eb_params.edge_weight),
            (
                content_frames[keyframe_idx],
                content_frames[target_idx],
                eb_params.image_weight,
            ),
            (source_pos_guide, target_pos_guide, eb_params.pos_weight),
            (style_img, warped_previous_style, eb_params.warp_weight),
        ]

    def _run_a_pass(
        self,
        seq: SynthesisSequence,
        style_img: np.ndarray,
        is_forward: bool,
        content_frames: List[np.ndarray],
        edge_maps: List[np.ndarray],
        fwd_flows: List[np.ndarray],
        modulation_frames: List[np.ndarray] | None,
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

        for source_idx in tqdm(frame_indices, desc=desc):
            target_idx = source_idx + step

            if is_forward:
                flow = fwd_flows[source_idx]
            else:
                flow = fwd_flows[target_idx]

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
                edge_maps=edge_maps,
                content_frames=content_frames,
            )

            modulation_map = (
                modulation_frames[target_idx] if modulation_frames else None
            )

            run_output = self.synthesis_engine.run(
                style_img,
                guides=guides,
                modulation_map=modulation_map,
                output_nnf=False,
            )
            stylized_img, err = run_output[0], run_output[1]

            stylized_frames.append(stylized_img)
            error_maps.append(err)

            if len(run_output) == 3:
                nnf_maps.append(run_output[2])

        if not is_forward:
            stylized_frames.reverse()
            error_maps.reverse()
            flows_used_in_pass.reverse()
            nnf_maps.reverse()

        return stylized_frames, error_maps, flows_used_in_pass, nnf_maps
