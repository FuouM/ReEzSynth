# comfyui_ezsynth/nodes/preprocess_nodes.py
"""
Preprocessing nodes for optical flow and edge detection.
"""

import hashlib
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import EZBaseNode


class OpticalFlowNode(EZBaseNode):
    """
    Compute optical flow between consecutive frames.
    """

    CATEGORY = "ReEzSynth/Preprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
            },
            "optional": {
                "engine": (["raft", "neuflow"], {"default": "raft"}),
                "model": (["sintel", "flyingchairs", "kitti"], {"default": "sintel"}),
                "cache_enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("FLOW_LIST",)
    RETURN_NAMES = ("flows",)
    FUNCTION = "compute_flow"

    def compute_flow(
        self,
        frames: List[torch.Tensor],
        engine: str = "raft",
        model: str = "sintel",
        cache_enabled: bool = True,
    ) -> Tuple[List[torch.Tensor]]:
        """
        Compute optical flow between consecutive frames.

        Args:
            frames: List of frame tensors
            engine: Flow engine to use
            model: Model variant
            cache_enabled: Whether to use caching

        Returns:
            Tuple containing list of flow tensors
        """
        from ..core.cache_manager import get_cache_manager
        from ..core.tensor_utils import flow_to_tensor, tensor_to_numpy

        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to compute flow")

        frame_np_list = [tensor_to_numpy(f) for f in frames]
        cache_mgr = get_cache_manager() if cache_enabled else None

        flows = []

        for i in range(len(frame_np_list) - 1):
            frame1 = frame_np_list[i]
            frame2 = frame_np_list[i + 1]

            # Check cache
            frame_hash = hashlib.md5(frame1.tobytes()).hexdigest()[:8]
            cache_key = f"{frame_hash}_{engine}_{model}"

            if cache_enabled and cache_mgr:
                cached = cache_mgr.load_cached_flow(cache_key)
                if cached is not None:
                    flows.append(flow_to_tensor(cached))
                    continue

            # Compute flow
            if engine == "raft":
                from ezsynth.engines.flow_engine import RAFTFlowEngine

                flow_engine = RAFTFlowEngine(model_name=model, arch="RAFT")
            else:
                from ezsynth.engines.flow_engine import NeuFlowEngine

                flow_engine = NeuFlowEngine(model_name=model)

            flow_result = flow_engine.compute([frame1, frame2])[0]

            if cache_enabled and cache_mgr:
                cache_mgr.save_cached_flow(cache_key, flow_result)

            flows.append(flow_to_tensor(flow_result))

        return (flows,)


class EdgeDetectNode(EZBaseNode):
    """
    Generate edge maps for guide-based synthesis.
    """

    CATEGORY = "ReEzSynth/Preprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
            },
            "optional": {
                "method": (["classic", "page", "pst"], {"default": "classic"}),
                "cache_enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("edges",)
    FUNCTION = "detect_edges"

    def detect_edges(
        self,
        frames: List[torch.Tensor],
        method: str = "classic",
        cache_enabled: bool = True,
    ) -> Tuple[List[torch.Tensor]]:
        """
        Generate edge maps for input frames.

        Args:
            frames: List of frame tensors
            method: Edge detection algorithm
            cache_enabled: Whether to use caching

        Returns:
            Tuple containing list of edge map tensors
        """
        from ezsynth.engines.edge_engine import EdgeEngine

        from ..core.cache_manager import get_cache_manager
        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        if len(frames) == 0:
            return ([],)

        frame_np_list = [tensor_to_numpy(f) for f in frames]
        cache_mgr = get_cache_manager() if cache_enabled else None

        edge_engine = EdgeEngine(method=method.upper())
        edges_np = edge_engine.compute(frame_np_list)

        del edge_engine

        tensors = [numpy_to_tensor(e) for e in edges_np]

        return (tensors,)


class SparseFeatureNode(EZBaseNode):
    """
    Generate sparse feature guides from tracked features.
    """

    CATEGORY = "ReEzSynth/Preprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
                "flows": ("FLOW_LIST",),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("sparse_guides",)
    FUNCTION = "compute_sparse"

    def compute_sparse(
        self, frames: List[torch.Tensor], flows: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor]]:
        """
        Generate sparse feature guides.

        Args:
            frames: List of frame tensors
            flows: List of flow tensors

        Returns:
            Tuple containing list of sparse guide tensors
        """
        from ezsynth.utils.feature_utils import (
            generate_tracked_features,
            render_gaussian_guide,
        )

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        if len(frames) == 0 or len(flows) == 0:
            return ([],)

        frame0_np = tensor_to_numpy(frames[0])
        flows_np = [tensor_to_numpy(f) for f in flows]

        tracked_points = generate_tracked_features(frame0_np, flows_np)

        h, w = frame0_np.shape[:2]
        sparse_guides = [render_gaussian_guide(h, w, pts) for pts in tracked_points]

        tensors = [numpy_to_tensor(g) for g in sparse_guides]

        return (tensors,)
