"""Single-direction synthesis pass execution."""

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..config import DebugConfig
from ..guide import GuideObject
from ..precompute import PrecomputeState
from ..utils.sequence_utils import SynthesisSequence
from ..utils.warp_utils import PositionalGuide, Warp
from .synthesis_engine import EbsynthEngine, PreparedSynthesisContext


class SynthesisPassRunner:
    """Execute one forward or reverse sequence using precomputed guides."""

    def __init__(
        self,
        *,
        engine: EbsynthEngine,
        precompute_state: PrecomputeState,
        debug_cfg: DebugConfig,
    ) -> None:
        self.engine = engine
        self.precompute_state = precompute_state
        self.debug_cfg = debug_cfg
        self._warp_tools_cache: dict[tuple[int, int], tuple[Warp, PositionalGuide]] = {}

    def run(
        self,
        *,
        seq: SynthesisSequence,
        style_img: np.ndarray,
        is_forward: bool,
        content_frames: List[np.ndarray],
        collect_intermediates: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
        warp, pos_guider = self._warp_tools(h, w)
        source_pos_guide = pos_guider.get_pristine_guide_uint8()

        previous_nnf = None
        use_propagation = self.engine.pipeline_config.use_temporal_nnf_propagation
        synthesis_context = None

        for source_idx in tqdm(frame_indices, desc=desc):
            target_idx = source_idx + step
            transition_idx = source_idx if is_forward else target_idx
            flow = self.precompute_state.fwd_flows[transition_idx]

            if collect_intermediates:
                flows_used_in_pass.append(flow)

            previous_stylized_frame = stylized_frames[-1]
            warped_previous_style = warp.run_warping(
                previous_stylized_frame,
                flow * (-step),
            )
            current_target_pos_guide = pos_guider.create_from_flow(flow)

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
                    previous_nnf.astype(np.float32),
                    flow * (-step),
                )
                initial_nnf_for_target = warped_nnf_float.astype(np.int32)

            if synthesis_context is None:
                synthesis_context = PreparedSynthesisContext(self.engine, style_img, guides)

            run_output = synthesis_context.run_frame(
                guides=guides,
                initial_nnf=initial_nnf_for_target,
                return_error=collect_intermediates,
                output_nnf=use_propagation,
            )

            if use_propagation:
                stylized_img, err, nnf = run_output
                previous_nnf = nnf
                if collect_intermediates:
                    nnf_maps.append(nnf)
            else:
                stylized_img, err = run_output

            stylized_frames.append(stylized_img)
            if collect_intermediates:
                error_maps.append(err)

        if not is_forward:
            stylized_frames.reverse()
            error_maps.reverse()
            flows_used_in_pass.reverse()
            nnf_maps.reverse()

        return stylized_frames, error_maps, flows_used_in_pass, nnf_maps

    def _warp_tools(self, height: int, width: int) -> tuple[Warp, PositionalGuide]:
        key = (height, width)
        tools = self._warp_tools_cache.get(key)
        if tools is None:
            use_taichi = self.engine.ebsynth_config.backend == "taichi"
            tools = (
                Warp(height, width, use_taichi=use_taichi),
                PositionalGuide(height, width, use_taichi=use_taichi),
            )
            self._warp_tools_cache[key] = tools
        return tools

    def _prepare_guides_for_frame(
        self,
        keyframe_idx: int,
        target_idx: int,
        style_img: np.ndarray,
        warped_previous_style: np.ndarray,
        source_pos_guide: np.ndarray,
        target_pos_guide: np.ndarray,
        content_frames: List[np.ndarray],
    ) -> List[GuideObject]:
        eb_params = self.engine.ebsynth_config
        guides = [
            GuideObject(
                keyframe=self.precompute_state.edge_maps[keyframe_idx],
                target=self.precompute_state.edge_maps[target_idx],
                weight=eb_params.edge_weight,
            ),
            GuideObject(
                keyframe=content_frames[keyframe_idx],
                target=content_frames[target_idx],
                weight=eb_params.image_weight,
            ),
            GuideObject(
                keyframe=source_pos_guide,
                target=target_pos_guide,
                weight=eb_params.pos_weight,
            ),
            GuideObject(
                keyframe=style_img,
                target=warped_previous_style,
                weight=eb_params.warp_weight,
            ),
        ]

        if (
            self.engine.pipeline_config.use_sparse_feature_guide
            and self.precompute_state.sparse_guides
        ):
            guides.append(
                GuideObject(
                    keyframe=self.precompute_state.sparse_guides[keyframe_idx],
                    target=self.precompute_state.sparse_guides[target_idx],
                    weight=eb_params.sparse_anchor_weight,
                )
            )
        return guides
