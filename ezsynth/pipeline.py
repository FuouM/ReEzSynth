from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import MainConfig
from .data import ProjectData
from .engines.edge_engine import EdgeEngine
from .engines.flow_engine import RAFTFlowEngine
from .engines.synthesis_engine import EbsynthEngine
from .pyramid import (
    build_laplacian_pyramid,
    downsample_flow,
    downsample_image,
    np_to_tensor,
    reconstruct_from_laplacian_pyramid,
    remap_image_to_residual,
    remap_residual_to_image,
    tensor_to_np,
)
from .utils.blend_utils import Blender
from .utils.sequence_utils import SynthesisSequence, create_sequences
from .utils.warp_utils import PositionalGuide, Warp


class SynthesisPipeline:
    """
    Orchestrates the entire video synthesis process, managing data, engines,
    and the core synthesis loop. Supports both single-pass and multi-pass
    pyramidal synthesis.
    """

    def __init__(self, config: MainConfig, data: ProjectData):
        self.config = config
        self.data = data

        # Engines are owned by the pipeline
        self.synthesis_engine = EbsynthEngine(config.ebsynth_params)
        self.flow_engine = RAFTFlowEngine(
            model_name=config.precomputation.flow_model,
            arch=config.precomputation.flow_engine,
        )
        self.edge_engine = EdgeEngine(method=config.precomputation.edge_method)

        # In-memory caches for pre-computed data
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
        Main entry point. Chooses synthesis strategy and runs optional final pass.
        """
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        style_frames = self.data.get_style_frames()
        self._compute_all_data(content_frames)

        # --- Main Synthesis Stage ---
        if (
            self.config.pipeline.pyramid_levels > 1
            and self.config.pipeline.use_residual_transfer
        ):
            print("\n--- Starting Pyramidal Synthesis with Residual Transfer ---")
            stable_frames = self._run_pyramidal_synthesis_residual(
                content_frames, style_frames
            )
        elif self.config.pipeline.pyramid_levels > 1:
            print("\n--- Starting Pyramidal Synthesis (Direct Upsampling) ---")
            stable_frames = self._run_pyramidal_synthesis_direct(
                content_frames, style_frames
            )
        else:
            print("\n--- Starting Single-Level Synthesis ---")
            stable_frames = self._run_single_level_synthesis(
                content_frames, style_frames, self._edge_maps, self._fwd_flows
            )

        # --- Final Pass Stage ---
        if self.config.pipeline.final_pass.enabled:
            print("\n--- Starting Final Stylization Pass ---")
            final_frames = self._run_final_pass(
                content_frames, style_frames, stable_frames
            )
        else:
            final_frames = stable_frames

        self.data.save_output_frames(final_frames)
        print("\nSynthesis pipeline finished.")

    def _run_final_pass(
        self,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
        stable_pyramid_output: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Performs a final stylization pass to re-inject detail and sharpness.
        Uses the stable pyramidal output as a strong guide.
        """
        final_frames = []
        style_indices = self.config.project.style_indices
        strength = self.config.pipeline.final_pass.strength
        base_img_weight = self.config.ebsynth_params.image_weight

        for i, content_frame in tqdm(
            enumerate(content_frames),
            total=len(content_frames),
            desc="Final Pass",
        ):
            # Find the nearest keyframe style to use for this frame
            nearest_style_idx = min(style_indices, key=lambda x: abs(x - i))
            # Map the frame index to the style_frames list index
            style_map_idx = style_indices.index(nearest_style_idx)
            style_img = style_frames[style_map_idx]

            # The source for the guides is the nearest keyframe
            source_content = content_frames[nearest_style_idx]
            source_stable = stable_pyramid_output[nearest_style_idx]

            # The target for the guides is the current frame
            target_content = content_frame
            target_stable = stable_pyramid_output[i]

            # We use the stable output as a very strong guide
            guides = [
                (source_stable, target_stable, base_img_weight * strength),
                (source_content, target_content, base_img_weight),
            ]

            # The "style" we are propagating is the original high-detail keyframe style
            stylized_img, _ = self.synthesis_engine.eb.run(style_img, guides=guides)
            final_frames.append(stylized_img)

        return final_frames

    def _run_pyramidal_synthesis_direct(
        self, content_frames: List[np.ndarray], style_frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Executes the coarse-to-fine pyramidal pipeline using direct upsampling."""
        num_levels = self.config.pipeline.pyramid_levels
        current_stylized_sequence = None

        for level in range(num_levels - 1, -1, -1):
            print(f"\n--- Processing Pyramid Level {level} (Res 1/{2**level}) ---")
            level_content = [downsample_image(f, level) for f in content_frames]
            level_styles = [downsample_image(s, level) for s in style_frames]
            level_edges = [downsample_image(e, level) for e in self._edge_maps]
            level_flows = [downsample_flow(f, level) for f in self._fwd_flows]

            if current_stylized_sequence is not None:
                h, w, _ = level_content[0].shape
                upsampled_input = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                    for img in current_stylized_sequence
                ]
                level_content = upsampled_input

            current_stylized_sequence = self._run_single_level_synthesis(
                level_content, level_styles, level_edges, level_flows
            )
        return current_stylized_sequence

    def _run_pyramidal_synthesis_residual(
        self, content_frames: List[np.ndarray], style_frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Executes pyramidal synthesis using Laplacian residuals for stability."""
        num_levels = self.config.pipeline.pyramid_levels

        print("Decomposing content frames into Laplacian pyramids...")
        content_pyramids = [
            build_laplacian_pyramid(frame, num_levels)
            for frame in tqdm(content_frames, desc="Decomposing")
        ]

        coarsest_level = num_levels - 1
        print(f"\n--- Processing Pyramid Level {coarsest_level} (Base) ---")
        level_content = [
            tensor_to_np(pyramid[coarsest_level]) for pyramid in content_pyramids
        ]
        level_styles = [downsample_image(s, coarsest_level) for s in style_frames]
        level_edges = [downsample_image(e, coarsest_level) for e in self._edge_maps]
        level_flows = [downsample_flow(f, coarsest_level) for f in self._fwd_flows]

        output_pyramid_base_np = self._run_single_level_synthesis(
            level_content, level_styles, level_edges, level_flows
        )

        output_pyramids = [
            [None] * (num_levels - 1) + [np_to_tensor(base)]
            for base in output_pyramid_base_np
        ]

        for level in range(num_levels - 2, -1, -1):
            print(f"\n--- Processing Pyramid Level {level} (Details) ---")

            # --- START OF FIX ---
            # The "style" to be synthesized is the remapped detail layer.
            detail_residuals = [pyramid[level] for pyramid in content_pyramids]
            style_frames_for_level = [
                remap_residual_to_image(res) for res in detail_residuals
            ]

            # The GUIDES are based on the stable, original downsampled content.
            content_for_guides = [downsample_image(f, level) for f in content_frames]
            edges_for_guides = [downsample_image(e, level) for e in self._edge_maps]
            flows_for_level = [downsample_flow(f, level) for f in self._fwd_flows]
            # --- END OF FIX ---

            stylized_detail_images = self._run_single_level_synthesis(
                content_frames=content_for_guides,
                style_frames=style_frames_for_level,  # Pass details as the 'style'
                edge_maps=edges_for_guides,
                fwd_flows=flows_for_level,
            )

            stylized_details_residuals = [
                remap_image_to_residual(img) for img in stylized_detail_images
            ]

            for i in range(len(output_pyramids)):
                output_pyramids[i][level] = stylized_details_residuals[i]

        print("\nReconstructing final frames from stylized pyramids...")
        final_frames = [
            reconstruct_from_laplacian_pyramid(pyramid)
            for pyramid in tqdm(output_pyramids, desc="Reconstructing")
        ]
        return final_frames

    def _run_single_level_synthesis(
        self,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
        edge_maps: List[np.ndarray],
        fwd_flows: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Executes the complete synthesis process (fwd, rev, blend) for a single
        resolution level. This is the reusable core logic.
        """
        style_indices = self.config.project.style_indices
        sequences = create_sequences(
            num_frames=len(content_frames), style_indices=style_indices
        )

        final_stylized_frames = []
        for seq_idx, seq in enumerate(sequences):
            is_not_first_sequence = seq_idx > 0

            # Pass the level-specific data to the pass runner
            pass_runner_args = {
                "seq": seq,
                "content_frames": content_frames,
                "edge_maps": edge_maps,
                "fwd_flows": fwd_flows,
            }

            if seq.mode == SynthesisSequence.MODE_FWD:
                style_img = style_frames[seq.style_indices[0]]
                styled_sequence, _, _ = self._run_a_pass(
                    **pass_runner_args, style_img=style_img, is_forward=True
                )
                if is_not_first_sequence:
                    styled_sequence.pop(0)
                final_stylized_frames.extend(styled_sequence)

            elif seq.mode == SynthesisSequence.MODE_REV:
                style_img = style_frames[seq.style_indices[0]]
                styled_sequence, _, _ = self._run_a_pass(
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

                fwd_frames, fwd_err, fwd_flows_used = self._run_a_pass(
                    **pass_runner_args, style_img=style_fwd, is_forward=True
                )
                bwd_frames, bwd_err, _ = self._run_a_pass(
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
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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

            stylized_img, err = self.synthesis_engine.eb.run(style_img, guides=guides)
            stylized_frames.append(stylized_img)
            error_maps.append(err)

        if not is_forward:
            stylized_frames.reverse()
            error_maps.reverse()
            flows_used_in_pass.reverse()

        return stylized_frames, error_maps, flows_used_in_pass
