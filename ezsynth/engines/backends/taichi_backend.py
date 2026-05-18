# ezsynth/engines/backends/taichi_backend.py
import platform
from typing import Optional, Tuple
import taichi as ti
import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import (
    COST_FUNCTION_NCC,
    EBSYNTH_VOTEMODE_PLAIN,
    TAICHI_INIT_VERBOSE,
)
from ...utils.timer import SynthesisTimer
from .base import BaseSynthesisBackend


# Initialize Taichi
def get_ti_arch():
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return ti.metal
    elif torch.cuda.is_available():
        return ti.cuda
    return ti.cpu


_ti_initialized = False


def ensure_ti_init():
    global _ti_initialized
    if not _ti_initialized:
        arch = get_ti_arch()
        ti.init(
            arch=arch,
            log_level=ti.INFO if TAICHI_INIT_VERBOSE else ti.ERROR,
            random_seed=42,
        )
        if TAICHI_INIT_VERBOSE:
            print(f"[Taichi] Initialized with arch: {arch}")
        _ti_initialized = True


@ti.data_oriented
class TaichiBackend(BaseSynthesisBackend):
    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        super().__init__(ebsynth_config, pipeline_config)
        ensure_ti_init()
        self.device = "cpu"
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

        self.timer = SynthesisTimer()
        self.benchmark_enabled = False

    def enable_benchmarking(self, enabled: bool = True):
        self.benchmark_enabled = enabled
        if enabled:
            self.timer.reset()

    def _timed_operation(self, operation_name: str, operation_func):
        if self.benchmark_enabled:
            with self.timer.time_operation(operation_name):
                return operation_func()
        else:
            return operation_func()

    # --- KERNELS AND FUNCS ---
    from .taichi_kernels import (
        compute_error_map_kernel,
        compute_integral_image,
        compute_patch_ncc,
        compute_patch_ssd,
        dilate_mask_kernel,
        eval_mask_kernel,
        get_omega,
        patchmatch_step_kernel,
        populate_omega,
        query_sat,
        random_search_kernel,
        voting_kernel,
    )

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
        if benchmark:
            self.enable_benchmarking(True)
        target_h, target_w = target_guide_tensor.shape[:2]
        source_h, source_w = style_tensor.shape[:2]
        orig_device = style_tensor.device

        def to_ti(t):
            return (
                t.contiguous() if orig_device.type == "cuda" else t.cpu().contiguous()
            )

        style_ti, s_guide_ti, t_guide_ti, modulation_ti = (
            to_ti(style_tensor),
            to_ti(source_guide_tensor),
            to_ti(target_guide_tensor),
            to_ti(modulation_tensor),
        )
        if modulation_ti.numel() == 0:
            modulation_ti = torch.zeros((1, 1, 1), dtype=torch.uint8)
        nnf_ti, s_weights_ti, g_weights_ti = (
            to_ti(nnf),
            to_ti(style_weights),
            to_ti(guide_weights),
        )
        error_map, omega_map, mask, mask2 = (
            torch.zeros((target_h, target_w), dtype=torch.float32),
            torch.zeros((source_h, source_w), dtype=torch.int32),
            torch.full((target_h, target_w), 255, dtype=torch.uint8),
            torch.zeros((target_h, target_w), dtype=torch.uint8),
        )
        output_image, target_style_prev = (
            torch.zeros((target_h, target_w, style_tensor.shape[2]), dtype=torch.uint8),
            torch.zeros((target_h, target_w, style_tensor.shape[2]), dtype=torch.uint8),
        )
        use_mod = 1 if modulation_tensor.numel() > 0 else 0
        omega_best = max(
            1e-6,
            (target_h * target_w) / (source_h * source_w) * (patch_size * patch_size),
        )

        use_bilateral = 1 if self.ebsynth_config.use_bilateral else 0
        sigma_spatial = self.ebsynth_config.sigma_spatial
        sigma_color = self.ebsynth_config.sigma_color
        n_size_step = self.ebsynth_config.n_size_step

        s_sat, s_sq_sat = (
            torch.zeros((source_h, source_w), dtype=torch.float32),
            torch.zeros((source_h, source_w), dtype=torch.float32),
        )
        t_sat, t_sq_sat = (
            torch.zeros((target_h, target_w), dtype=torch.float32),
            torch.zeros((target_h, target_w), dtype=torch.float32),
        )
        if cost_function_mode == COST_FUNCTION_NCC:
            self._timed_operation(
                "source_sats",
                lambda: (
                    self.compute_integral_image(style_ti, s_sat, 0),
                    self.compute_integral_image(style_ti, s_sq_sat, 1),
                ),
            )

        self._timed_operation(
            "populate_omega", lambda: self.populate_omega(nnf_ti, omega_map)
        )
        self._timed_operation(
            "initial_vote",
            lambda: self.voting_kernel(
                target_style_prev,
                style_ti,
                target_style_prev,  # Dummy target style for center
                nnf_ti,
                error_map,
                patch_size,
                EBSYNTH_VOTEMODE_PLAIN,
                use_bilateral,
                sigma_spatial,
                sigma_color,
                n_size_step,
            ),
        )

        for iter_idx in range(search_vote_iters):
            if cost_function_mode == COST_FUNCTION_NCC:
                self._timed_operation(
                    f"target_sats_{iter_idx}",
                    lambda: (
                        self.compute_integral_image(target_style_prev, t_sat, 0),
                        self.compute_integral_image(target_style_prev, t_sq_sat, 1),
                    ),
                )
            self._timed_operation(
                f"error_map_{iter_idx}",
                lambda: self.compute_error_map_kernel(
                    nnf_ti,
                    error_map,
                    style_ti,
                    target_style_prev,
                    s_guide_ti,
                    t_guide_ti,
                    modulation_ti,
                    use_mod,
                    patch_size,
                    s_weights_ti,
                    g_weights_ti,
                    cost_function_mode,
                    s_sat,
                    s_sq_sat,
                    t_sat,
                    t_sq_sat,
                    use_bilateral,
                    sigma_spatial,
                    sigma_color,
                    n_size_step,
                ),
            )
            for pm_idx in range(patch_match_iters):
                self._timed_operation(
                    f"pm_step_{iter_idx}_{pm_idx}",
                    lambda: self.patchmatch_step_kernel(
                        nnf_ti,
                        error_map,
                        omega_map,
                        style_ti,
                        target_style_prev,
                        s_guide_ti,
                        t_guide_ti,
                        modulation_ti,
                        use_mod,
                        s_weights_ti,
                        g_weights_ti,
                        patch_size,
                        pm_idx % 2,
                        uniformity_weight,
                        mask,
                        cost_function_mode,
                        omega_best,
                        s_sat,
                        s_sq_sat,
                        t_sat,
                        t_sq_sat,
                        use_bilateral,
                        sigma_spatial,
                        sigma_color,
                        n_size_step,
                    ),
                )

            self._timed_operation(
                f"random_search_{iter_idx}",
                lambda: self.random_search_kernel(
                    nnf_ti,
                    error_map,
                    omega_map,
                    style_ti,
                    target_style_prev,
                    s_guide_ti,
                    t_guide_ti,
                    modulation_ti,
                    use_mod,
                    s_weights_ti,
                    g_weights_ti,
                    patch_size,
                    max(source_w, source_h) // 2,
                    uniformity_weight,
                    mask,
                    self.ebsynth_config.search_pruning_threshold,
                    cost_function_mode,
                    omega_best,
                    s_sat,
                    s_sq_sat,
                    t_sat,
                    t_sq_sat,
                    use_bilateral,
                    sigma_spatial,
                    sigma_color,
                    n_size_step,
                ),
            )

            self._timed_operation(
                f"vote_{iter_idx}",
                lambda: self.voting_kernel(
                    output_image,
                    style_ti,
                    target_style_prev,
                    nnf_ti,
                    error_map,
                    patch_size,
                    vote_mode,
                    use_bilateral,
                    sigma_spatial,
                    sigma_color,
                    n_size_step,
                ),
            )

            if iter_idx < search_vote_iters - 1:
                self.eval_mask_kernel(
                    mask, output_image, target_style_prev, int(stop_threshold)
                )
                self.dilate_mask_kernel(mask2, mask, patch_size)
                mask.copy_(mask2)
            target_style_prev.copy_(output_image)

        if orig_device.type == "cuda":
            return (
                output_image.to("cuda"),
                error_map.to("cuda"),
                nnf_ti.to("cuda"),
            )
        return output_image, error_map, nnf_ti
