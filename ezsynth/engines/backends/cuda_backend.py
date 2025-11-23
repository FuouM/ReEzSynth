# ezsynth/engines/backends/cuda_backend.py
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import CUDA_EXTENSION_AVAILABLE, ebsynth_torch
from ...torch_ops import SynthesisTimer
from .base import BaseSynthesisBackend


class CudaBackend(BaseSynthesisBackend):
    """
    CUDA backend using the native ebsynth_torch extension.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        super().__init__(ebsynth_config, pipeline_config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend selected but CUDA is not available.")
        if not CUDA_EXTENSION_AVAILABLE:
            raise RuntimeError(
                "CUDA backend selected but ebsynth_torch extension is not available."
            )
        self.device = "cuda"
        self.timer = SynthesisTimer()
        self.benchmark_enabled = False

    def enable_benchmarking(self, enabled: bool = True):
        """Enable or disable detailed benchmarking."""
        self.benchmark_enabled = enabled
        if enabled:
            self.timer.reset()

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
        Run a single level using the CUDA extension.
        """
        if benchmark:
            self.enable_benchmarking(True)

        if self.benchmark_enabled:
            with self.timer.time_operation("cuda_level_execution"):
                return ebsynth_torch.run_level(
                    style_tensor,
                    source_guide_tensor,
                    target_guide_tensor,
                    modulation_tensor,
                    nnf,
                    style_weights,
                    guide_weights,
                    uniformity_weight,
                    patch_size,
                    vote_mode,
                    search_vote_iters,
                    patch_match_iters,
                    stop_threshold,
                    rand_states,
                    self.ebsynth_config.search_pruning_threshold,
                    cost_function_mode,
                )
        else:
            return ebsynth_torch.run_level(
                style_tensor,
                source_guide_tensor,
                target_guide_tensor,
                modulation_tensor,
                nnf,
                style_weights,
                guide_weights,
                uniformity_weight,
                patch_size,
                vote_mode,
                search_vote_iters,
                patch_match_iters,
                stop_threshold,
                rand_states,
                self.ebsynth_config.search_pruning_threshold,
                cost_function_mode,
            )
