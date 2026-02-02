# ezsynth/engines/backends/cuda_backend.py
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import EXTENSION_AVAILABLE, EXTENSION_CUDA_AVAILABLE, ebsynth_torch
from ...utils import SynthesisTimer
from .base import BaseSynthesisBackend


class CudaBackend(BaseSynthesisBackend):
    """
    CUDA backend using the native ebsynth_torch extension.
    Also supports CPU tensors through the unified dispatch layer.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig, device: str = None
    ):
        super().__init__(ebsynth_config, pipeline_config)
        if not EXTENSION_AVAILABLE:
            raise RuntimeError(
                "Extension backend selected but ebsynth_torch extension is not available."
            )
        # Device can be specified or auto-detected
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timer = SynthesisTimer()
        self.benchmark_enabled = False
        self._warned_cpu_fallback = False

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
        Run a single level using the extension (auto-detects CPU/CUDA).
        """
        # Check if we're running on CPU and warn once
        if style_tensor.device.type == "cpu" and not self._warned_cpu_fallback:
            if not EXTENSION_CUDA_AVAILABLE:
                print("[INFO] Running on CPU (CUDA not available in extension).")
            else:
                print("[INFO] Running on CPU (input tensors are on CPU).")
            self._warned_cpu_fallback = True

        if benchmark:
            self.enable_benchmarking(True)

        device_type = "cpu" if style_tensor.device.type == "cpu" else "cuda"
        timer_name = f"{device_type}_level_execution"

        if self.benchmark_enabled:
            with self.timer.time_operation(timer_name):
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
