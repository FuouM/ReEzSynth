# comfyui_ezsynth/types.py
"""
Custom type definitions for ComfyUI nodes.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import numpy as np
import torch

# Type aliases for clarity
ImageTensor = torch.Tensor  # Shape: (H, W, C), dtype: uint8/float32
FlowTensor = torch.Tensor  # Shape: (H, W, 2), dtype: float32
NNFTensor = torch.Tensor  # Shape: (H, W, 2), dtype: int32

GuideTuple = Tuple[ImageTensor, float]  # (guide_image, weight)
GuideList = List[GuideTuple]

# FaceBlit types
Landmarks = List[Tuple[int, int]]  # 68 facial landmarks


@dataclass
class FaceBlitAssets:
    """FaceBlit style assets bundle."""

    style_path: str
    landmarks_path: str
    pos_guide_path: str
    app_guide_path: str
    lut_path: str

    def to_dict(self) -> dict:
        return {
            "style_path": self.style_path,
            "landmarks_path": self.landmarks_path,
            "pos_guide_path": self.pos_guide_path,
            "app_guide_path": self.app_guide_path,
            "lut_path": self.lut_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FaceBlitAssets":
        return cls(**data)


@dataclass
class EbsynthParams:
    """Ebsynth parameters bundle."""

    uniformity: float = 3500.0
    patch_size: int = 7
    pyramid_levels: int = 6
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    backend: str = "cuda"
    extra_pass_3x3: bool = False
    cost_function: str = "ssd"
    # Guide weights
    edge_weight: float = 1.0
    image_weight: float = 6.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5
    sparse_anchor_weight: float = 50.0


@dataclass
class SequenceConfig:
    """Sequence configuration for video processing."""

    start_frame: int
    end_frame: int
    style_index: int
    mode: str  # "forward", "reverse", "blend"


# ComfyUI-compatible type mappings
class EZTypes:
    """Type mappings for ComfyUI node system."""

    IMAGE = "IMAGE"  # Standard ComfyUI image (B, H, W, C)
    IMAGE_LIST = "IMAGE_LIST"  # List of images for sequences
    FLOW_LIST = "FLOW_LIST"  # List of optical flow tensors
    FLOW = "FLOW"  # Single optical flow tensor
    NNF = "NNF"  # Nearest neighbor field
    NNF_LIST = "NNF_LIST"  # List of NNFs
    GUIDE = "GUIDE"  # Single guide with weight
    GUIDE_LIST = "GUIDE_LIST"  # List of guides
    EB_PARAMS = "EB_PARAMS"  # Ebsynth parameter bundle
    SEQUENCE = "SEQUENCE"  # Sequence configuration
    SEQUENCE_LIST = "SEQUENCE_LIST"
    # FaceBlit types
    LANDMARKS = "LANDMARKS"  # 68 facial landmarks
    FB_ASSETS = "FB_ASSETS"  # FaceBlit style assets
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"


# Custom output type classes for ComfyUI
class IMAGE_LIST:
    """Custom type for list of images."""

    @staticmethod
    def get_name():
        return "IMAGE_LIST"

    @staticmethod
    def get_choices():
        return None


class FLOW_LIST:
    """Custom type for list of optical flows."""

    @staticmethod
    def get_name():
        return "FLOW_LIST"

    @staticmethod
    def get_choices():
        return None


class FLOW:
    """Custom type for single optical flow."""

    @staticmethod
    def get_name():
        return "FLOW"

    @staticmethod
    def get_choices():
        return None


class NNF_LIST:
    """Custom type for list of NNFs."""

    @staticmethod
    def get_name():
        return "NNF_LIST"

    @staticmethod
    def get_choices():
        return None


class GUIDE_LIST:
    """Custom type for list of guides."""

    @staticmethod
    def get_name():
        return "GUIDE_LIST"

    @staticmethod
    def get_choices():
        return None


class LANDMARKS:
    """Custom type for facial landmarks."""

    @staticmethod
    def get_name():
        return "LANDMARKS"

    @staticmethod
    def get_choices():
        return None


class FB_ASSETS:
    """Custom type for FaceBlit assets."""

    @staticmethod
    def get_name():
        return "FB_ASSETS"

    @staticmethod
    def get_choices():
        return None


class EB_PARAMS:
    """Custom type for Ebsynth parameters."""

    @staticmethod
    def get_name():
        return "EB_PARAMS"

    @staticmethod
    def get_choices():
        return None


class SEQUENCE_LIST:
    """Custom type for sequence list."""

    @staticmethod
    def get_name():
        return "SEQUENCE_LIST"

    @staticmethod
    def get_choices():
        return None
