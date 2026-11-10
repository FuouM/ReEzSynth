# ezsynth/engines/backends/pytorch_backend.py

import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import (
    EBSYNTH_VOTEMODE_WEIGHTED,
    SEARCH_MODE_BY_NAME,
    SEARCH_MODE_JFA,
    SEARCH_MODE_MULTICAND,
)
from ...torch_ops.device_cache import clear_torch_device_cache
from ...torch_ops.jfa_ops import jfa_search
from ...torch_ops.mask_ops import dilate_mask, evaluate_mask
from ...torch_ops.multicand_ops import (
    CandidateSet,
    initial_candidate_set,
    propagation_step_multicand,
    refresh_and_promote,
)
from ...torch_ops.omega_ops import populate_omega_map
from ...torch_ops.patch_memory import LevelShape, PatchPlan, plan_level
from ...torch_ops.patch_ops import ImagePatches, extract_patches
from ...torch_ops.patchmatch_ops import (
    propagation_step,
    random_search_step,
    try_patch_batch,
)
from ...torch_ops.voting_ops import vote_plain, vote_weighted
from ...utils.timer import SynthesisTimer
from .common import get_auto_torch_device, resample_tensor

logger = logging.getLogger(__name__)

PATCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class _LevelSearch:
    """Everything the correspondence search of one pyramid level reads or updates."""

    nnf: torch.Tensor
    omega_map: torch.Tensor
    omega_best: float
    source_style_patches: torch.Tensor | ImagePatches
    source_guide_patches: torch.Tensor | ImagePatches
    target_guide_patches: torch.Tensor | ImagePatches
    target_modulation_patches: torch.Tensor | ImagePatches | None
    style_weights: torch.Tensor
    guide_weights: torch.Tensor
    uniformity_weight: float
    patch_size: int
    cost_function_mode: int
    search_pruning_threshold: float
    random_generator: torch.Generator | None
    cost_kw: dict
    alternatives: CandidateSet | None


class PyTorchBackend:
    """
    PyTorch backend with CPU/GPU support using torch_ops.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        self.ebsynth_config = ebsynth_config
        self.pipeline_config = pipeline_config
        self.device = get_auto_torch_device()
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
        self.search_mode = SEARCH_MODE_BY_NAME[ebsynth_config.search_mode]
        self.patch_dtype = PATCH_DTYPES[ebsynth_config.precision]

        self.timer = SynthesisTimer()
        self.benchmark_enabled = False
        self._announced_plans: set[tuple] = set()

    def enable_benchmarking(self, enabled: bool = True):
        """Enable or disable detailed benchmarking."""
        self.benchmark_enabled = enabled
        if enabled:
            self.timer.reset()

    def _timed(self, operation_name: str):
        """Context manager timing ``operation_name`` when benchmarking is enabled."""
        if self.benchmark_enabled:
            return self.timer.time_operation(operation_name)
        return nullcontext()

    def _patch_grid(
        self,
        tensor: torch.Tensor,
        patch_size: int,
        plan: PatchPlan,
        *,
        clear_device_cache: bool = True,
    ) -> torch.Tensor | ImagePatches:
        """Unfold grid for the materialized plan, on-demand view for the bounded one.

        Grids are stored in the configured precision; the cost kernels compute in
        float32 (uint8 images are exact in float16/bfloat16).
        """
        if plan.chunk_pixels is None:
            return extract_patches(
                tensor,
                patch_size,
                as_float=True,
                clear_device_cache=clear_device_cache,
                dtype=self.patch_dtype,
            )
        return ImagePatches(tensor, patch_size, plan.chunk_pixels, self.patch_dtype)

    def _plan_level(
        self,
        style_tensor: torch.Tensor,
        source_guide_tensor: torch.Tensor,
        target_guide_tensor: torch.Tensor,
        modulation_tensor: torch.Tensor,
        patch_size: int,
        *,
        iterative: bool,
    ) -> PatchPlan:
        shape = LevelShape(
            source_hw=tuple(style_tensor.shape[:2]),
            target_hw=tuple(target_guide_tensor.shape[:2]),
            style_channels=style_tensor.shape[2],
            guide_channels=source_guide_tensor.shape[2],
            modulation_channels=(
                modulation_tensor.shape[2] if modulation_tensor.numel() > 0 else 0
            ),
            patch_size=patch_size,
            patch_element_bytes=self.patch_dtype.itemsize,
            candidate_slots=self._candidate_slots(),
        )
        plan = plan_level(shape, style_tensor.device, iterative=iterative)
        key = (shape, plan.strategy, plan.chunk_pixels)
        if plan.strategy == "bounded" and key not in self._announced_plans:
            self._announced_plans.add(key)
            budget = (
                f"{plan.budget_bytes >> 20} MiB"
                if plan.budget_bytes is not None
                else "unknown"
            )
            logger.info(
                "[torch backend] level %dx%d: materialized patches need ~%d MiB "
                "(budget %s); gathering patches in chunks of %d pixels",
                shape.target_hw[1],
                shape.target_hw[0],
                plan.estimate_bytes >> 20,
                budget,
                plan.chunk_pixels,
            )
        return plan

    def _candidate_slots(self) -> int:
        if self.search_mode != SEARCH_MODE_MULTICAND:
            return 0
        return self.ebsynth_config.multicand_k - 1

    def _make_random_generator(self, device: torch.device) -> torch.Generator | None:
        """Return a deterministic generator when PyTorch supports one for the device."""
        device_type = torch.device(device).type
        if device_type == "mps":
            # MPS torch.randint does not consistently accept explicit generators.
            return None
        generator = torch.Generator(device=device_type)
        generator.manual_seed(1337)
        return generator

    def _cost_kw(self) -> dict:
        return {
            "use_bilateral": self.ebsynth_config.use_bilateral,
            "sigma_spatial": self.ebsynth_config.sigma_spatial,
            "sigma_color": self.ebsynth_config.sigma_color,
            "n_size_step": self.ebsynth_config.n_size_step,
        }

    def _vote_kw(self) -> dict:
        return {
            "use_bilateral": self.ebsynth_config.use_bilateral,
            "sigma_spatial": self.ebsynth_config.sigma_spatial,
            "sigma_color": self.ebsynth_config.sigma_color,
        }

    # --- Correspondence search -------------------------------------------------

    def _propagate(
        self, search: _LevelSearch, target_style_patches, error_map, mask, is_odd
    ) -> None:
        args = (
            search.nnf,
            error_map,
            search.omega_map,
            search.source_style_patches,
            target_style_patches,
            search.source_guide_patches,
            search.target_guide_patches,
            search.style_weights,
            search.guide_weights,
            search.uniformity_weight,
            search.patch_size,
            is_odd,
            mask,
            search.cost_function_mode,
            search.omega_best,
        )
        if search.alternatives is not None:
            propagation_step_multicand(
                search.alternatives,
                *args,
                target_modulation_patches=search.target_modulation_patches,
                **search.cost_kw,
            )
        else:
            propagation_step(
                *args,
                target_modulation_patches=search.target_modulation_patches,
                **search.cost_kw,
            )

    def _random_search(
        self, search: _LevelSearch, target_style_patches, error_map, mask, radius
    ) -> None:
        random_search_step(
            search.nnf,
            error_map,
            search.omega_map,
            search.source_style_patches,
            target_style_patches,
            search.source_guide_patches,
            search.target_guide_patches,
            search.style_weights,
            search.guide_weights,
            search.uniformity_weight,
            search.patch_size,
            radius,
            mask,
            search.search_pruning_threshold,
            search.cost_function_mode,
            search.omega_best,
            generator=search.random_generator,
            target_modulation_patches=search.target_modulation_patches,
            alternatives=search.alternatives,
            **search.cost_kw,
        )

    def _jfa(self, search: _LevelSearch, target_style_patches, error_map, mask) -> None:
        jfa_search(
            search.nnf,
            error_map,
            search.omega_map,
            search.source_style_patches,
            target_style_patches,
            search.source_guide_patches,
            search.target_guide_patches,
            search.style_weights,
            search.guide_weights,
            search.uniformity_weight,
            search.patch_size,
            mask,
            search.cost_function_mode,
            search.omega_best,
            search.search_pruning_threshold,
            generator=search.random_generator,
            target_modulation_patches=search.target_modulation_patches,
            **search.cost_kw,
        )

    def _search_phase(
        self,
        search: _LevelSearch,
        target_style_patches,
        error_map,
        mask,
        patch_match_iters: int,
        radius: int,
    ) -> None:
        """
        One full search pass.

        PatchMatch / multicand (matches ``dispatch.cu`` / ``dispatch_cpu.cpp`` /
        Taichi): ``patch_match_iters`` propagation-only steps, then one random-search
        step -- not interleaved; pass direction matches ``(i % 2 == 1)`` on the CUDA
        side. JFA: one jump-flooding sweep (all step sizes) + fine random search,
        independent of ``patch_match_iters``.
        """
        if self.search_mode == SEARCH_MODE_JFA:
            with self._timed("jfa_search"):
                self._jfa(search, target_style_patches, error_map, mask)
            return
        for iteration in range(patch_match_iters):
            with self._timed("propagation_step"):
                self._propagate(
                    search, target_style_patches, error_map, mask, iteration % 2 == 1
                )
        with self._timed("random_search_step"):
            self._random_search(search, target_style_patches, error_map, mask, radius)

    def _vote(self, style_tensor, search: _LevelSearch, error_map, vote_mode):
        if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
            alternatives = (
                (search.alternatives.nnf, search.alternatives.errors)
                if search.alternatives is not None
                else None
            )
            return vote_weighted(
                style_tensor,
                search.nnf,
                error_map,
                search.patch_size,
                alternatives=alternatives,
                **self._vote_kw(),
            )
        return vote_plain(
            style_tensor, search.nnf, search.patch_size, **self._vote_kw()
        )

    def _current_errors(
        self, search: _LevelSearch, scratch_errors_inf, target_style_patches
    ) -> torch.Tensor:
        """Patch cost of the current NNF against ``target_style_patches``."""
        return try_patch_batch(
            search.nnf,
            search.nnf,
            scratch_errors_inf,
            search.omega_map,
            search.source_style_patches,
            target_style_patches,
            search.source_guide_patches,
            search.target_guide_patches,
            search.style_weights,
            search.guide_weights,
            search.uniformity_weight,
            search.patch_size,
            search.cost_function_mode,
            search.omega_best,
            target_modulation_patches=search.target_modulation_patches,
            **search.cost_kw,
        )[1]

    def _initial_alternatives(self, nnf: torch.Tensor) -> CandidateSet | None:
        if self.search_mode != SEARCH_MODE_MULTICAND:
            return None
        return initial_candidate_set(nnf, self.ebsynth_config.multicand_k)

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
        Run a single level using PyTorch operations.
        """
        if benchmark:
            self.enable_benchmarking(True)

        random_generator = self._make_random_generator(nnf.device)
        run = (
            self._run_level_pytorch_iterative
            if self.pipeline_config.use_residual_transfer
            else self._run_level_pytorch
        )
        return run(
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
            cost_function_mode,
            random_generator,
        )

    def _run_level_pytorch(
        self,
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
        cost_function_mode,
        random_generator,
    ):
        """
        Synthesis level with a fixed target style (no residual transfer): one
        search phase, ``search_vote_iters`` extra random searches (JFA: sweeps),
        then a single vote.
        """
        H_s, W_s, _C_s = style_tensor.shape
        H_t, W_t, _C_g = target_guide_tensor.shape

        # Pre-compute all patch grids once before the loops (unfold grids, or
        # on-demand views when the unfold grids would not fit the memory budget).
        plan = self._plan_level(
            style_tensor,
            source_guide_tensor,
            target_guide_tensor,
            modulation_tensor,
            patch_size,
            iterative=False,
        )

        with self._timed("patch_extraction"):
            source_style_patches = self._patch_grid(
                style_tensor, patch_size, plan, clear_device_cache=False
            )
            if H_s == H_t and W_s == W_t:
                target_style_patches = source_style_patches
            else:
                target_style_patches = self._patch_grid(
                    resample_tensor(style_tensor, H_t, W_t),
                    patch_size,
                    plan,
                    clear_device_cache=False,
                )
            source_guide_patches = self._patch_grid(
                source_guide_tensor, patch_size, plan, clear_device_cache=False
            )
            has_mod = modulation_tensor.numel() > 0
            target_guide_patches = self._patch_grid(
                target_guide_tensor, patch_size, plan, clear_device_cache=not has_mod
            )
            target_modulation_patches = (
                self._patch_grid(modulation_tensor, patch_size, plan)
                if has_mod
                else None
            )

        with self._timed("omega_initialization"):
            omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
            error_map = torch.full(
                (H_t, W_t), float("inf"), dtype=torch.float32, device=self.device
            )
        clear_torch_device_cache(self.device)

        search = _LevelSearch(
            nnf=nnf,
            omega_map=omega_map,
            omega_best=(H_t * W_t) / (H_s * W_s),
            source_style_patches=source_style_patches,
            source_guide_patches=source_guide_patches,
            target_guide_patches=target_guide_patches,
            target_modulation_patches=target_modulation_patches,
            style_weights=style_weights,
            guide_weights=guide_weights,
            uniformity_weight=uniformity_weight,
            patch_size=patch_size,
            cost_function_mode=cost_function_mode,
            search_pruning_threshold=self.ebsynth_config.search_pruning_threshold,
            random_generator=random_generator,
            cost_kw=self._cost_kw(),
            alternatives=self._initial_alternatives(nnf),
        )
        # Patch grids are referenced through ``search`` from here on.
        del source_style_patches, source_guide_patches, target_guide_patches
        del target_modulation_patches

        with self._timed("nnf_initialization"):
            # ``try_patch_batch`` treats candidate/current coordinates as read-only.
            # Its result is not assigned: the search starts from an all-inf error
            # map, as in the CUDA dispatch.
            try_patch_batch(
                nnf,
                nnf,
                error_map,
                omega_map,
                search.source_style_patches,
                target_style_patches,
                search.source_guide_patches,
                search.target_guide_patches,
                style_weights,
                guide_weights,
                uniformity_weight,
                patch_size,
                cost_function_mode,
                search.omega_best,
                target_modulation_patches=search.target_modulation_patches,
                **search.cost_kw,
            )

        convergence_mask = torch.full(
            (H_t, W_t), 255, dtype=torch.uint8, device=self.device
        )
        full_radius = max(H_s, W_s) // 2

        with self._timed("patchmatch_iterations"):
            self._search_phase(
                search,
                target_style_patches,
                error_map,
                convergence_mask,
                patch_match_iters,
                full_radius,
            )

        # Additional search-vote iterations (the target style is fixed here, so
        # only the exploration step repeats).
        with self._timed("search_vote_iterations"):
            for _ in range(search_vote_iters):
                if self.search_mode == SEARCH_MODE_JFA:
                    with self._timed("jfa_search"):
                        self._jfa(
                            search, target_style_patches, error_map, convergence_mask
                        )
                else:
                    with self._timed("random_search_step"):
                        self._random_search(
                            search,
                            target_style_patches,
                            error_map,
                            convergence_mask,
                            full_radius,
                        )

        with self._timed("voting"):
            output_image = self._vote(style_tensor, search, error_map, vote_mode)

        del search, target_style_patches
        clear_torch_device_cache(self.device)

        return output_image, error_map, nnf

    def _run_level_pytorch_iterative(
        self,
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
        cost_function_mode,
        random_generator,
    ):
        """
        New, fast algorithm matching CUDA. Iteratively refines the NNF and the
        synthesized image together in a loop.
        """
        H_s, W_s, _ = style_tensor.shape
        H_t, W_t, _ = target_guide_tensor.shape
        vote_kw = self._vote_kw()

        # Pre-extract all source patches once, as they don't change (unfold grids,
        # or on-demand views when the unfold grids would not fit the memory budget).
        plan = self._plan_level(
            style_tensor,
            source_guide_tensor,
            target_guide_tensor,
            modulation_tensor,
            patch_size,
            iterative=True,
        )

        with self._timed("patch_extraction"):
            source_style_patches = self._patch_grid(
                style_tensor, patch_size, plan, clear_device_cache=False
            )
            source_guide_patches = self._patch_grid(
                source_guide_tensor, patch_size, plan, clear_device_cache=False
            )
            # The target guide does not change during the iterative process.
            has_mod = modulation_tensor.numel() > 0
            target_guide_patches = self._patch_grid(
                target_guide_tensor, patch_size, plan, clear_device_cache=not has_mod
            )
            target_modulation_patches = (
                self._patch_grid(modulation_tensor, patch_size, plan)
                if has_mod
                else None
            )

        with self._timed("omega_initialization"):
            omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
            omega_best = max((H_t * W_t) / (H_s * W_s), 1e-6)
        clear_torch_device_cache(self.device)

        search = _LevelSearch(
            nnf=nnf,
            omega_map=omega_map,
            omega_best=omega_best,
            source_style_patches=source_style_patches,
            source_guide_patches=source_guide_patches,
            target_guide_patches=target_guide_patches,
            target_modulation_patches=target_modulation_patches,
            style_weights=style_weights,
            guide_weights=guide_weights,
            uniformity_weight=uniformity_weight,
            patch_size=patch_size,
            cost_function_mode=cost_function_mode,
            search_pruning_threshold=self.ebsynth_config.search_pruning_threshold,
            random_generator=random_generator,
            cost_kw=self._cost_kw(),
            alternatives=self._initial_alternatives(nnf),
        )
        # Patch grids are referenced through ``search`` from here on.
        del source_style_patches, source_guide_patches, target_guide_patches
        del target_modulation_patches

        # Match cpu/dispatch_cpu.cpp: initial vote uses an all-zero error map (extension
        # allocates output_error with zeros). Do not run try_patch before this vote.
        target_style_temp = vote_plain(style_tensor, nnf, patch_size, **vote_kw)

        # One buffer for the previous frame; reuse to avoid a full HxWxC clone each
        # iteration.
        target_style_prev = torch.empty_like(target_style_temp)
        target_style_prev.copy_(target_style_temp)
        mask = torch.full((H_t, W_t), 255, dtype=torch.uint8, device=self.device)

        # Reused for every try_patch_batch that starts from an all-inf error map
        # (read-only input).
        scratch_errors_inf = torch.full(
            (H_t, W_t), float("inf"), dtype=torch.float32, device=self.device
        )
        full_radius = max(H_s, W_s) // 2
        with self._timed("iterative_refinement"):
            for iteration in range(search_vote_iters):
                with self._timed("target_patch_extraction"):
                    target_style_patches = self._patch_grid(
                        target_style_prev, patch_size, plan
                    )
                with self._timed("nnf_update"):
                    error_map = self._current_errors(
                        search, scratch_errors_inf, target_style_patches
                    )
                if search.alternatives is not None:
                    with self._timed("multicand_refresh"):
                        refresh_and_promote(
                            search.alternatives,
                            nnf,
                            error_map,
                            omega_map,
                            search.source_style_patches,
                            target_style_patches,
                            search.source_guide_patches,
                            search.target_guide_patches,
                            style_weights,
                            guide_weights,
                            uniformity_weight,
                            patch_size,
                            mask,
                            cost_function_mode,
                            omega_best,
                            target_modulation_patches=search.target_modulation_patches,
                            **search.cost_kw,
                        )
                with self._timed("patchmatch_iterations"):
                    self._search_phase(
                        search,
                        target_style_patches,
                        error_map,
                        mask,
                        patch_match_iters,
                        full_radius,
                    )
                # Free the target patches before voting.
                del target_style_patches

                with self._timed("voting"):
                    target_style_temp = self._vote(
                        style_tensor, search, error_map, vote_mode
                    )

                if iteration < search_vote_iters - 1 and stop_threshold > 0:
                    with self._timed("mask_evaluation"):
                        new_mask = evaluate_mask(
                            target_style_temp, target_style_prev, stop_threshold
                        )
                        mask = dilate_mask(new_mask, patch_size)

                target_style_prev.copy_(target_style_temp)

        if search_vote_iters == 0:
            target_style_temp = target_style_prev

        with self._timed("final_output"):
            output_error = self._current_errors(
                search,
                scratch_errors_inf,
                self._patch_grid(target_style_temp, patch_size, plan),
            )

        del search, scratch_errors_inf, target_style_prev
        clear_torch_device_cache(self.device)

        return target_style_temp, output_error, nnf
