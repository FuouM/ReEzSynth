# ezsynth/engines/backends/wgpu_backend.py
"""WebGPU backend: the ``ezsynth_wgpu`` Rust module (``rust/ezsynth-wgpu``).

Runs on Vulkan, DX12 or Metal through wgpu; any vendor's GPU works and no CUDA toolkit
is needed. Level inputs stay on the CPU (the engine's device is ``"cpu"``) and are
uploaded per level; the kernels follow the Taichi backend.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import (
    COST_FUNCTION_NCC,
    COST_FUNCTION_SAD,
    COST_FUNCTION_SSD,
    EBSYNTH_VOTEMODE_PLAIN,
    EBSYNTH_VOTEMODE_WEIGHTED,
)
from ...utils.timer import SynthesisTimer

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The 'wgpu' backend needs the ezsynth-wgpu package. Install the wheel for your OS "
    "from https://github.com/FuouM/ReEzSynth/releases with `pip install <wheel>`, or "
    "build it from the repository with `pip install ./rust/ezsynth-wgpu` / "
    "`uv sync --extra wgpu` (requires a Rust toolchain)."
)

_COST_NAMES = {
    COST_FUNCTION_SSD: "ssd",
    COST_FUNCTION_NCC: "ncc",
    COST_FUNCTION_SAD: "sad",
}
_VOTE_NAMES = {EBSYNTH_VOTEMODE_WEIGHTED: "weighted", EBSYNTH_VOTEMODE_PLAIN: "plain"}

_SYNTHESIZER = None


def import_wgpu():
    """Import ``ezsynth_wgpu`` or raise ImportError with install instructions."""
    try:
        import ezsynth_wgpu
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return ezsynth_wgpu


def shared_synthesizer():
    """Process-wide synthesizer (one WebGPU device, cached pipelines)."""
    global _SYNTHESIZER
    if _SYNTHESIZER is None:
        _SYNTHESIZER = import_wgpu().Synthesizer()
        logger.info(f"wgpu backend on {_SYNTHESIZER.adapter}")
    return _SYNTHESIZER


def _array(t: torch.Tensor, dtype) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype)


class WgpuBackend:
    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        self.ebsynth_config = ebsynth_config
        self.pipeline_config = pipeline_config
        self._synth = shared_synthesizer()
        self.device = "cpu"
        self.timer = SynthesisTimer()
        self.benchmark_enabled = False
        # Random stream per call: levels and frames get distinct streams, runs repeat.
        self._calls = 0

    def enable_benchmarking(self, enabled: bool = True):
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
        rand_states: torch.Tensor | None,
        cost_function_mode: int,
        benchmark: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if benchmark:
            self.enable_benchmarking(True)
        cfg = self.ebsynth_config
        modulation = (
            _array(modulation_tensor, np.uint8)
            if modulation_tensor.numel() > 0
            else None
        )
        if modulation is not None and modulation.shape[2] != guide_weights.numel():
            raise ValueError(
                f"modulation has {modulation.shape[2]} channels; the guides have "
                f"{guide_weights.numel()}"
            )
        seed = self._calls
        self._calls += 1

        def run():
            return self._synth.run_level(
                _array(style_tensor, np.uint8),
                _array(source_guide_tensor, np.uint8),
                _array(target_guide_tensor, np.uint8),
                modulation,
                _array(nnf, np.int32),
                _array(style_weights, np.float32),
                _array(guide_weights, np.float32),
                patch_size=int(patch_size),
                uniformity=float(uniformity_weight),
                vote_mode=_VOTE_NAMES[vote_mode],
                search_vote_iters=int(search_vote_iters),
                patch_match_iters=int(patch_match_iters),
                stop_threshold=int(stop_threshold),
                search_pruning_threshold=float(cfg.search_pruning_threshold),
                cost_function=_COST_NAMES[cost_function_mode],
                search_mode=cfg.search_mode,
                multicand_k=int(cfg.multicand_k),
                use_bilateral=bool(cfg.use_bilateral),
                sigma_spatial=float(cfg.sigma_spatial),
                sigma_color=float(cfg.sigma_color),
                n_size_step=int(cfg.n_size_step),
                seed=seed,
            )

        if self.benchmark_enabled:
            with self.timer.time_operation("wgpu_level_execution"):
                image, error, nnf_out = run()
        else:
            image, error, nnf_out = run()
        return (
            torch.from_numpy(image),
            torch.from_numpy(error),
            torch.from_numpy(nnf_out),
        )
