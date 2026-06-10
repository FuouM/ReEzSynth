"""Single-direction synthesis pass execution."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..config import DebugConfig
from ..guide import GuideObject
from ..precompute import PrecomputeState
from ..utils.io_utils import write_image
from ..utils.occlusion import (
    composite_masked_regions,
    inpaint_masked_regions,
    mask_to_bgr,
    modulation_from_mask,
)
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
        pipeline_config = self.engine.pipeline_config
        synthesis_context = None

        for source_idx in tqdm(frame_indices, desc=desc):
            target_idx = source_idx + step
            transition_idx = source_idx if is_forward else target_idx
            flow = self.precompute_state.fwd_flows[transition_idx]
            directed_flow = flow * (-step)
            occlusion_masks = (
                self.precompute_state.fwd_occlusion_masks
                if is_forward
                else self.precompute_state.bwd_occlusion_masks
            )
            occlusion_mask = None
            if occlusion_masks and 0 <= transition_idx < len(occlusion_masks):
                occlusion_mask = occlusion_masks[transition_idx]

            if collect_intermediates:
                flows_used_in_pass.append(flow)

            s2t_flow = flow if is_forward else -flow
            previous_stylized_frame = stylized_frames[-1]
            if pipeline_config.use_forward_warping:
                warped_previous_style = warp.run_forward_warping(
                    previous_stylized_frame,
                    s2t_flow,
                )
            else:
                warped_previous_style = warp.run_warping(
                    previous_stylized_frame,
                    directed_flow,
                )
            current_target_pos_guide = pos_guider.create_from_flow(
                s2t_flow if pipeline_config.use_forward_warping else flow
            )

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
                warped_nnf_float = warp.run_warping_float_map(
                    previous_nnf,
                    directed_flow,
                    interpolation=cv2.INTER_NEAREST,
                )
                initial_nnf_for_target = warped_nnf_float.astype(np.int32, copy=False)

            modulation_map = None
            if (
                pipeline_config.use_flow_occlusion_modulation
                and occlusion_mask is not None
            ):
                modulation_map = modulation_from_mask(
                    occlusion_mask,
                    floor=pipeline_config.occlusion_modulation_floor,
                )
            self._save_pass_occlusion_debug(
                is_forward=is_forward,
                target_idx=target_idx,
                mask=occlusion_mask,
                modulation_map=modulation_map,
                fill_enabled=pipeline_config.use_flow_occlusion_fill,
            )

            if synthesis_context is None:
                synthesis_context = PreparedSynthesisContext(
                    self.engine, style_img, guides
                )

            run_output = synthesis_context.run_frame(
                guides=guides,
                modulation_map=modulation_map,
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

            if pipeline_config.use_flow_occlusion_fill and occlusion_mask is not None:
                stylized_img = inpaint_masked_regions(
                    stylized_img,
                    occlusion_mask,
                    radius=pipeline_config.occlusion_fill_radius,
                    feather=pipeline_config.occlusion_fill_feather,
                )

            if pipeline_config.use_flow_occlusion_refine:
                stylized_img = self._refine_occluded_regions(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    source_style_img=previous_stylized_frame,
                    base_stylized_img=stylized_img,
                    occlusion_mask=occlusion_mask,
                    content_frames=content_frames,
                )

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
            # Keep the default pass on OpenCV remap; the Taichi warp path is only
            # needed for explicit forward splatting.
            use_taichi = self.engine.pipeline_config.use_forward_warping
            tools = (
                Warp(height, width, use_taichi=use_taichi),
                PositionalGuide(
                    height,
                    width,
                    use_taichi=use_taichi,
                    use_forward_warp=self.engine.pipeline_config.use_forward_warping,
                ),
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

    def _refine_occluded_regions(
        self,
        *,
        source_idx: int,
        target_idx: int,
        source_style_img: np.ndarray,
        base_stylized_img: np.ndarray,
        occlusion_mask: Optional[np.ndarray],
        content_frames: List[np.ndarray],
    ) -> np.ndarray:
        if occlusion_mask is None or not np.any(occlusion_mask):
            return base_stylized_img

        eb_params = self.engine.ebsynth_config
        guides = [
            GuideObject(
                keyframe=content_frames[source_idx],
                target=content_frames[target_idx],
                weight=eb_params.image_weight,
            ),
            GuideObject(
                keyframe=self.precompute_state.edge_maps[source_idx],
                target=self.precompute_state.edge_maps[target_idx],
                weight=eb_params.edge_weight,
            ),
            GuideObject(
                keyframe=source_style_img,
                target=base_stylized_img,
                weight=eb_params.warp_weight,
            ),
        ]

        refined_img, _ = self.engine.run(
            source_style_img,
            guides=guides,
            return_error=False,
            output_nnf=False,
        )
        return composite_masked_regions(
            base_stylized_img,
            refined_img,
            occlusion_mask,
            feather=self.engine.pipeline_config.occlusion_refine_feather,
        )

    def _save_pass_occlusion_debug(
        self,
        *,
        is_forward: bool,
        target_idx: int,
        mask: Optional[np.ndarray],
        modulation_map: Optional[np.ndarray],
        fill_enabled: bool,
    ) -> None:
        if not self.debug_cfg.save_occlusion_debug or mask is None:
            return

        pass_dir = self._occlusion_debug_output_dir() / (
            "pass_fwd" if is_forward else "pass_bwd"
        )
        masks_dir = pass_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        write_image(masks_dir / f"{target_idx:05d}.png", mask_to_bgr(mask))

        if modulation_map is not None:
            mod_dir = pass_dir / "modulation"
            mod_dir.mkdir(parents=True, exist_ok=True)
            write_image(mod_dir / f"{target_idx:05d}.png", modulation_map)

        if fill_enabled:
            fill_dir = pass_dir / "final_composite_masks"
            fill_dir.mkdir(parents=True, exist_ok=True)
            write_image(fill_dir / f"{target_idx:05d}.png", mask_to_bgr(mask))

    def _occlusion_debug_output_dir(self) -> Path:
        p = Path(self.debug_cfg.occlusion_debug_dir)
        return p if p.is_absolute() else Path.cwd() / p
