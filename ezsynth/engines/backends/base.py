# ezsynth/engines/backends/base.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from .common import random_init_nnf, resample_tensor


class BaseSynthesisBackend(ABC):
    """
    Abstract base class for synthesis backends.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        self.ebsynth_config = ebsynth_config
        self.pipeline_config = pipeline_config
        self.device = None  # To be set by subclasses

    @abstractmethod
    def run_level(
        self,
        style_tensor: torch.Tensor,
        source_guide_tensor: torch.Tensor,
        target_guide_tensor: torch.Tensor,
        modulation_tensor: torch.Tensor,
        nnf: torch.Tensor,
        style_weights: torch.Tensor,
        guide_weights: torch.Tensor,
        uniformity_weight: float,
        patch_size: int,
        vote_mode: int,
        search_vote_iters: int,
        patch_match_iters: int,
        stop_threshold: float,
        rand_states: Optional[torch.Tensor],
        cost_function_mode: int,
        benchmark: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a single level of the synthesis algorithm.

        Returns:
            Tuple of (output_image, output_error, nnf)
        """
        pass

    def _resample_tensor(
        self, tensor: torch.Tensor, new_h: int, new_w: int, mode: str = "bilinear"
    ) -> torch.Tensor:
        """Helper to resample a tensor using torch.nn.functional.interpolate."""
        return resample_tensor(tensor, new_h, new_w, mode)

    def _init_nnf(self, target_h, target_w, source_h, source_w, patch_size):
        """Initializes a random NNF on the GPU."""
        return random_init_nnf(
            self.device, target_h, target_w, source_h, source_w, patch_size
        )
