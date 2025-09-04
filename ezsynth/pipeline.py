# ezsynth/pipeline.py
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .config import MainConfig
from .data import ProjectData

# Engines are imported just-in-time to save memory
from .engines.synthesis_engine import EbsynthEngine
from .utils.blend_utils import Blender
from .utils.feature_utils import generate_tracked_features, render_gaussian_guide
from .utils.io_utils import load_frames_from_dir, write_image
from .utils.sequence_utils import SynthesisSequence, create_sequences
from .utils.warp_utils import PositionalGuide, Warp


class SynthesisPipeline:
    """
    Orchestrates the entire video synthesis process. Manages a memory-safe,
    sequential pre-computation workflow with caching, and then delegates to the
    core synthesis engine for the main processing loop.
    """

    def __init__(self, config: MainConfig, data: ProjectData):
        self.config = config
        self.data = data

        # The synthesis engine is used repeatedly, so we initialize it here.
        # It's a C++ extension and manages its own memory efficiently.
        self.synthesis_engine = EbsynthEngine(
            ebsynth_config=config.ebsynth_params, pipeline_config=config.pipeline
        )

        # Pre-computation results will be stored here after they are computed or loaded
        self._edge_maps: List[np.ndarray] = []
        self._fwd_flows: List[np.ndarray] = []
        self._sparse_guides: List[np.ndarray] = []

    def _compute_optical_flow(self, content_frames: List[np.ndarray]):
        """Load or compute optical flow, ensuring the model is cleared from memory afterwards."""
        print("\n--- Pre-computation: Optical Flow ---")
        cache_dir = Path(self.config.project.cache_dir) / "flow"
        num_expected_flows = len(content_frames) - 1

        if (
            not self.config.project.force_recompute_flow
            and cache_dir.exists()
            and len(list(cache_dir.glob("*.npy"))) == num_expected_flows
        ):
            print(f"Loading optical flow from cache: {cache_dir}")
            flow_paths = sorted(cache_dir.glob("*.npy"))
            self._fwd_flows = [
                np.load(p) for p in tqdm(flow_paths, desc="Loading Cached Flow")
            ]
        else:
            from .engines.flow_engine import (  # Just-in-time import
                NeuFlowEngine,
                RAFTFlowEngine,
            )

            print("Instantiating Flow Engine...")
            engine_name = self.config.precomputation.flow_engine.upper()
            if engine_name == "RAFT":
                engine = RAFTFlowEngine(
                    model_name=self.config.precomputation.flow_model, arch=engine_name
                )
            elif engine_name == "NEUFLOW":
                engine = NeuFlowEngine(model_name=self.config.precomputation.flow_model)
            else:
                raise ValueError(f"Unknown flow engine: '{engine_name}'")

            self._fwd_flows = engine.compute(content_frames)
            print(f"Saving {len(self._fwd_flows)} flow fields to cache: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            for i, flow in enumerate(tqdm(self._fwd_flows, desc="Saving Flow Cache")):
                np.save(cache_dir / f"{i:05d}.npy", flow)

            print("Optical flow computation complete. Releasing model from memory...")
            del engine
            torch.cuda.empty_cache()

        print("Optical flow pre-computation finished.")

    def _compute_edge_maps(self, content_frames: List[np.ndarray]):
        """Load or compute edge maps."""
        print("\n--- Pre-computation: Edge Maps ---")
        edge_method_name = self.config.precomputation.edge_method.lower()
        cache_dir = Path(self.config.project.cache_dir) / f"edges_{edge_method_name}"

        if (
            not self.config.project.force_recompute_edge
            and cache_dir.exists()
            and len(list(cache_dir.glob("*.png"))) == len(content_frames)
        ):
            print(f"Loading edge maps from cache: {cache_dir}")
            self._edge_maps = load_frames_from_dir(cache_dir)
        else:
            from .engines.edge_engine import EdgeEngine  # Just-in-time import

            engine = EdgeEngine(method=self.config.precomputation.edge_method)
            self._edge_maps = engine.compute(content_frames)

            print(f"Saving {len(self._edge_maps)} edge maps to cache: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            for i, edge_map in enumerate(self._edge_maps):
                write_image(cache_dir / f"{i:05d}.png", edge_map)
            del engine

        print("Edge map pre-computation finished.")

    def _compute_all_data(self, content_frames: List[np.ndarray]):
        """Runs all pre-computation steps sequentially and with memory management."""

        # 1. Optical Flow (VRAM intensive)
        self._compute_optical_flow(content_frames)

        # 2. Edge Maps (less intensive)
        self._compute_edge_maps(content_frames)

        # 3. Sparse Features (depends on flow, low VRAM)
        if self.config.pipeline.use_sparse_feature_guide:
            print("\nGenerating sparse feature guides...")
            tracked_points = generate_tracked_features(
                content_frames[0], self._fwd_flows
            )
            h, w, _ = content_frames[0].shape
            self._sparse_guides = [
                render_gaussian_guide(h, w, pts) for pts in tracked_points
            ]

        print("\nAll pre-computation finished.")

    def run(self) -> List[np.ndarray]:
        """
        Main entry point for the synthesis pipeline.
        """
        print("Loading project data for pipeline...")
        content_frames = self.data.get_content_frames()
        self.data.get_style_frames()  # Ensure styles are loaded and resized if needed

        # This will run the sequential pre-computation and populate the internal result lists
        self._compute_all_data(content_frames)

        print("\n--- Starting Synthesis ---")
        final_frames = self._run_synthesis(
            content_frames,
            self.data.get_style_frames(),  # Pass the loaded styles
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
        content_frames: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Prepares the list of guide tuples for a single Ebsynth run."""
        eb_params = self.config.ebsynth_params
        guides = [
            (
                self._edge_maps[keyframe_idx],
                self._edge_maps[target_idx],
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

        if self.config.pipeline.use_sparse_feature_guide and self._sparse_guides:
            guides.append(
                (
                    self._sparse_guides[keyframe_idx],
                    self._sparse_guides[target_idx],
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
        use_propagation = self.config.pipeline.use_temporal_nnf_propagation

        for source_idx in tqdm(frame_indices, desc=desc):
            target_idx = source_idx + step

            if is_forward:
                flow = self._fwd_flows[source_idx]
            else:
                flow = self._fwd_flows[target_idx]

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
