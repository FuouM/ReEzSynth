"""Synthesis on the native ``ebsynth_torch`` C++/CUDA extension (``cuda_src/``).

The extension also runs CPU tensors (OpenMP C++); ``cuda_loader`` imports the wheel or
JIT-builds it.
"""

import logging
from contextlib import nullcontext

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...utils.timer import SynthesisTimer
from . import cuda_loader as native

logger = logging.getLogger(__name__)


class Backend:
    """
    CUDA backend using the native ebsynth_torch extension.
    Also supports CPU tensors through the unified dispatch layer.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        self.ebsynth_config = ebsynth_config
        self.pipeline_config = pipeline_config
        if not native.ensure_extension_loaded():
            raise RuntimeError(
                "Extension backend selected but ebsynth_torch extension is not available."
            )
        self._ext = native.module
        # ``device`` in the config selects CPU or CUDA tensors; else CUDA when present.
        if ebsynth_config.device is not None:
            self.device = ebsynth_config.device
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

    def init_rand_states(self, rand_states: torch.Tensor) -> None:
        """Initialize native RNG state for the tensor's device."""
        self._ext.init_rand_states(rand_states)

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
        rand_states: torch.Tensor | None,
        cost_function_mode: int,
        benchmark: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a single level using the extension (auto-detects CPU/CUDA).
        """
        # Check if we're running on CPU and warn once
        if style_tensor.device.type == "cpu" and not self._warned_cpu_fallback:
            if not native.cuda_available:
                logger.info("[INFO] Running on CPU (CUDA not available in extension).")
            else:
                logger.info("[INFO] Running on CPU (input tensors are on CPU).")
            self._warned_cpu_fallback = True

        if benchmark:
            self.enable_benchmarking(True)

        device_type = "cpu" if style_tensor.device.type == "cpu" else "cuda"
        timed = (
            self.timer.time_operation(f"{device_type}_level_execution")
            if self.benchmark_enabled
            else nullcontext()
        )
        cfg = self.ebsynth_config
        with timed:
            return self._ext.run_level(
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
                cfg.search_pruning_threshold,
                cost_function_mode,
                cfg.use_optimization,
                cfg.use_bilateral,
                cfg.sigma_spatial,
                cfg.sigma_color,
                cfg.n_size_step,
            )
