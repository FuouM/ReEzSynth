# comfyui_ezsynth/__init__.py
"""
ReEzSynth ComfyUI Custom Nodes

This module provides ComfyUI custom nodes for the ReEzSynth video/image
synthesis pipeline, including Ebsynth, FastBlend, and FaceBlit integrations.
"""

from .nodes.faceblit_nodes import (
    FaceBlitGuideNode,
    FaceBlitStyleNode,
    FaceBlitStylizeNode,
    FaceDetectNode,
)
from .nodes.fastblend_nodes import FastBlendInterpolateNode, FastBlendNode
from .nodes.guide_nodes import CombineGuidesNode, PositionalGuideNode, WarpGuideNode
from .nodes.input_nodes import LoadFlowNode, LoadImageNode, LoadVideoNode
from .nodes.output_nodes import PreviewImageNode, SaveImageNode, SaveVideoNode
from .nodes.preprocess_nodes import EdgeDetectNode, OpticalFlowNode, SparseFeatureNode
from .nodes.sequence_nodes import BlendSequencesNode, ForwardPassNode, ReversePassNode
from .nodes.synthesis_nodes import EbsynthNode, ImageSynthNode
from .nodes.utility_nodes import ColorTransferNode, MaskNode, WarpImageNode

NODE_CLASS_MAPPINGS = {
    # I/O nodes
    "EZS_LoadVideo": LoadVideoNode,
    "EZS_LoadImage": LoadImageNode,
    "EZS_LoadFlow": LoadFlowNode,
    "EZS_SaveVideo": SaveVideoNode,
    "EZS_PreviewImage": PreviewImageNode,
    "EZS_SaveImage": SaveImageNode,
    # Preprocessing nodes
    "EZS_OpticalFlow": OpticalFlowNode,
    "EZS_EdgeDetect": EdgeDetectNode,
    "EZS_SparseFeatures": SparseFeatureNode,
    # Guide nodes
    "EZS_PositionalGuide": PositionalGuideNode,
    "EZS_WarpGuide": WarpGuideNode,
    "EZS_CombineGuides": CombineGuidesNode,
    # Synthesis nodes
    "EZS_ImageSynth": ImageSynthNode,
    "EZS_Ebsynth": EbsynthNode,
    # Sequence nodes
    "EZS_ForwardPass": ForwardPassNode,
    "EZS_ReversePass": ReversePassNode,
    "EZS_BlendSequences": BlendSequencesNode,
    # FastBlend nodes
    "EZS_FastBlend": FastBlendNode,
    "EZS_FastBlendInterpolate": FastBlendInterpolateNode,
    # FaceBlit nodes
    "EZS_FaceBlitStyle": FaceBlitStyleNode,
    "EZS_FaceBlitStylize": FaceBlitStylizeNode,
    "EZS_FaceDetect": FaceDetectNode,
    "EZS_FaceBlitGuide": FaceBlitGuideNode,
    # Utility nodes
    "EZS_WarpImage": WarpImageNode,
    "EZS_Mask": MaskNode,
    "EZS_ColorTransfer": ColorTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # I/O nodes
    "EZS_LoadVideo": "EZ Load Video",
    "EZS_LoadImage": "EZ Load Image",
    "EZS_LoadFlow": "EZ Load Flow",
    "EZS_SaveVideo": "EZ Save Video",
    "EZS_PreviewImage": "EZ Preview Image",
    "EZS_SaveImage": "EZ Save Image",
    # Preprocessing nodes
    "EZS_OpticalFlow": "EZ Optical Flow",
    "EZS_EdgeDetect": "EZ Edge Detection",
    "EZS_SparseFeatures": "EZ Sparse Features",
    # Guide nodes
    "EZS_PositionalGuide": "EZ Positional Guide",
    "EZS_WarpGuide": "EZ Warp Guide",
    "EZS_CombineGuides": "EZ Combine Guides",
    # Synthesis nodes
    "EZS_ImageSynth": "EZ Image Synthesis (Simple)",
    "EZS_Ebsynth": "EZ Ebsynth (Advanced)",
    # Sequence nodes
    "EZS_ForwardPass": "EZ Forward Pass",
    "EZS_ReversePass": "EZ Reverse Pass",
    "EZS_BlendSequences": "EZ Blend Sequences",
    # FastBlend nodes
    "EZS_FastBlend": "EZ FastBlend Smooth",
    "EZS_FastBlendInterpolate": "EZ FastBlend Interp",
    # FaceBlit nodes
    "EZS_FaceBlitStyle": "FB Style Assets",
    "EZS_FaceBlitStylize": "FB Stylize",
    "EZS_FaceDetect": "FB Face Detect",
    "EZS_FaceBlitGuide": "FB Guide Gen",
    # Utility nodes
    "EZS_WarpImage": "EZ Warp Image",
    "EZS_Mask": "EZ Mask",
    "EZS_ColorTransfer": "EZ Color Transfer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
