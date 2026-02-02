# comfyui_ezsynth/nodes/sequence_nodes.py
"""
Sequence processing nodes for video synthesis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..types import EbsynthParams
from .base import EZBaseNode


class ForwardPassNode(EZBaseNode):
    """
    Process a sequence of frames in forward direction.
    """

    CATEGORY = "ReEzSynth/Sequence"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
                "style_image": ("IMAGE",),
                "start_idx": ("INT", {"default": 0, "min": 0}),
                "end_idx": ("INT", {"default": 10, "min": 1}),
                "flows": ("FLOW_LIST",),
            },
            "optional": {
                "edges": ("IMAGE_LIST",),
                "use_temporal_nnf": ("BOOLEAN", {"default": True}),
                "uniformity": ("FLOAT", {"default": 3500.0}),
                "patch_size": ("INT", {"default": 7}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST", "IMAGE_LIST", "NNF_LIST")
    RETURN_NAMES = ("stylized_frames", "error_maps", "nnfs")
    FUNCTION = "run_forward_pass"

    def run_forward_pass(
        self,
        frames: List[torch.Tensor],
        style_image: torch.Tensor,
        start_idx: int,
        end_idx: int,
        flows: List[torch.Tensor],
        edges: Optional[List[torch.Tensor]] = None,
        use_temporal_nnf: bool = True,
        uniformity: float = 3500.0,
        patch_size: int = 7,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run forward synthesis pass.
        """
        from ezsynth.config import EbsynthParamsConfig, FinalPassConfig, PipelineConfig
        from ezsynth.engines.synthesis_engine import EbsynthEngine
        from ezsynth.utils.warp_utils import PositionalGuide, Warp

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_flow, tensor_to_numpy
        from ..types import SequenceConfig
        from ..utils.sequence_utils import SynthesisSequence

        frame_np_list = [tensor_to_numpy(f) for f in frames]
        flow_np_list = [tensor_to_flow(fl) for fl in flows]
        style_np = tensor_to_numpy(style_image)

        h, w = frame_np_list[0].shape[:2]

        # Initialize engine
        config = EbsynthParamsConfig(
            uniformity=uniformity,
            patch_size=patch_size,
            search_vote_iters=12,
            patch_match_iters=6,
        )
        pipeline_cfg = PipelineConfig(
            pyramid_levels=6,
            use_temporal_nnf_propagation=use_temporal_nnf,
            use_sparse_feature_guide=False,
        )
        engine = EbsynthEngine(ebsynth_config=config, pipeline_config=pipeline_cfg)

        warp = Warp(h, w)
        pos_guider = PositionalGuide(h, w)
        source_pos_guide = pos_guider.get_pristine_guide_uint8()

        stylized_frames = [style_np]
        error_maps = []
        nnf_maps = []
        previous_nnf = None

        for source_idx in range(start_idx, min(end_idx, len(frame_np_list) - 1)):
            target_idx = source_idx + 1
            flow = flow_np_list[source_idx]

            previous_stylized = stylized_frames[-1]
            warped_style = warp.run_warping(previous_stylized, flow * -1)
            target_pos_guide = pos_guider.create_from_flow(flow)

            # Build guides
            guides = [
                (style_np, warped_style, 0.5),
                (source_pos_guide, target_pos_guide, 2.0),
            ]

            if edges is not None and len(edges) > source_idx:
                edge_src = tensor_to_numpy(edges[start_idx])
                edge_tgt = tensor_to_numpy(edges[target_idx])
                guides.append((edge_src, edge_tgt, 1.0))

            # Initial NNF
            initial_nnf = None
            if use_temporal_nnf and previous_nnf is not None:
                warped_nnf = warp.run_warping(
                    previous_nnf.astype(np.float32), flow * -1
                )
                initial_nnf = warped_nnf.astype(np.int32)

            # Run synthesis
            if use_temporal_nnf:
                stylized, err, nnf = engine.run(
                    style_np, guides, initial_nnf=initial_nnf, output_nnf=True
                )
                previous_nnf = nnf
                nnf_maps.append(nnf)
            else:
                stylized, err = engine.run(style_np, guides, output_nnf=False)

            stylized_frames.append(stylized)
            error_maps.append(err)

        return (
            [numpy_to_tensor(f) for f in stylized_frames],
            [numpy_to_tensor(e) for e in error_maps],
            [numpy_to_tensor(n) for n in nnf_maps],
        )


class ReversePassNode(EZBaseNode):
    """
    Process a sequence of frames in reverse direction.
    """

    CATEGORY = "ReEzSynth/Sequence"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
                "style_image": ("IMAGE",),
                "start_idx": ("INT", {"default": 10, "min": 1}),
                "end_idx": ("INT", {"default": 0, "min": 0}),
                "flows": ("FLOW_LIST",),
            },
            "optional": {
                "edges": ("IMAGE_LIST",),
                "use_temporal_nnf": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST", "IMAGE_LIST", "NNF_LIST")
    RETURN_NAMES = ("stylized_frames", "error_maps", "nnfs")
    FUNCTION = "run_reverse_pass"

    def run_reverse_pass(
        self,
        frames: List[torch.Tensor],
        style_image: torch.Tensor,
        start_idx: int,
        end_idx: int,
        flows: List[torch.Tensor],
        edges: Optional[List[torch.Tensor]] = None,
        use_temporal_nnf: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run reverse synthesis pass.
        """
        # Similar to forward pass but in reverse direction
        # Implementation would mirror ForwardPassNode

        # Placeholder - actual implementation would be similar
        return ([], [], [])


class BlendSequencesNode(EZBaseNode):
    """
    Blend forward and backward pass results.
    """

    CATEGORY = "ReEzSynth/Sequence"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fwd_frames": ("IMAGE_LIST",),
                "bwd_frames": ("IMAGE_LIST",),
                "fwd_errors": ("IMAGE_LIST",),
                "bwd_errors": ("IMAGE_LIST",),
                "flows": ("FLOW_LIST",),
            },
            "optional": {
                "solver": (["lsqr", "cg"], {"default": "lsqr"}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("blended_frames",)
    FUNCTION = "blend_sequences"

    def blend_sequences(
        self,
        fwd_frames: List[torch.Tensor],
        bwd_frames: List[torch.Tensor],
        fwd_errors: List[torch.Tensor],
        bwd_errors: List[torch.Tensor],
        flows: List[torch.Tensor],
        solver: str = "lsqr",
    ) -> Tuple[List[torch.Tensor]]:
        """
        Blend forward and backward passes using error-based selection.
        """
        from ezsynth.config import BlendingConfig

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_flow, tensor_to_numpy
        from ..utils.blend_utils import Blender

        fwd_np = [tensor_to_numpy(f) for f in fwd_frames]
        bwd_np = [tensor_to_numpy(b) for b in bwd_frames]
        fwd_err_np = [tensor_to_numpy(e) for e in fwd_errors]
        bwd_err_np = [tensor_to_numpy(e) for e in bwd_errors]
        flow_np = [tensor_to_flow(fl) for fl in flows]

        h, w = fwd_np[0].shape[:2]

        blend_config = BlendingConfig(use_lsqr=(solver == "lsqr"))
        blender = Blender(h, w, **blend_config.model_dump())

        blended = blender.run(
            fwd_frames=fwd_np,
            bwd_frames=bwd_np,
            fwd_errors=fwd_err_np,
            bwd_errors=bwd_err_np,
            fwd_flows=flow_np,
        )

        return ([numpy_to_tensor(f) for f in blended],)
