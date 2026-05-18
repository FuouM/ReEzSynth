# ezsynth/engines/backends/pytorch_backend.py
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import EBSYNTH_VOTEMODE_WEIGHTED
from ...torch_ops.device_cache import clear_torch_device_cache
from ...torch_ops.mask_ops import dilate_mask, evaluate_mask
from ...torch_ops.omega_ops import populate_omega_map
from ...torch_ops.patchmatch_ops import propagation_step, random_search_step, try_patch_batch
from ...torch_ops.voting_ops import vote_plain, vote_weighted
from ...utils.timer import SynthesisTimer
from ...torch_ops.patch_ops import extract_patches
from .common import get_auto_torch_device, resample_tensor


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

        self.timer = SynthesisTimer()
        self.benchmark_enabled = False

    def enable_benchmarking(self, enabled: bool = True):
        """Enable or disable detailed benchmarking."""
        self.benchmark_enabled = enabled
        if enabled:
            self.timer.reset()

    def _timed_operation(self, operation_name: str, operation_func):
        """Execute an operation with optional timing."""
        if self.benchmark_enabled:
            with self.timer.time_operation(operation_name):
                return operation_func()
        else:
            return operation_func()

    @staticmethod
    def _extract_patches(
        tensor: torch.Tensor,
        patch_size: int,
        *,
        clear_device_cache: bool = True,
    ) -> torch.Tensor:
        # The torch backend immediately evaluates patch distances in float.
        # Keeping patch buffers float avoids repeated uint8->float casts in SSD/NCC.
        return extract_patches(
            tensor,
            patch_size,
            as_float=True,
            clear_device_cache=clear_device_cache,
        )

    def _make_random_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Return a deterministic generator when PyTorch supports one for the device."""
        device_type = torch.device(device).type
        if device_type == "mps":
            # MPS torch.randint does not consistently accept explicit generators.
            return None
        generator = torch.Generator(device=device_type)
        generator.manual_seed(1337)
        return generator

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
        Run a single level using PyTorch operations.
        """
        if benchmark:
            self.enable_benchmarking(True)

        random_generator = self._make_random_generator(nnf.device)

        if self.pipeline_config.use_residual_transfer:
            return self._run_level_pytorch_iterative(
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
        else:
            return self._run_level_pytorch(
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
        PyTorch implementation of the synthesis level using torch_ops.

        PatchMatch scheduling matches ``dispatch.cu`` / ``dispatch_cpu.cpp`` /
        Taichi: all propagation substeps run first, then one random-search step
        (not interleaved). Pass direction matches ``(i % 2 == 1)`` on the CUDA side.
        """
        H_s, W_s, C_s = style_tensor.shape
        H_t, W_t, C_g = target_guide_tensor.shape
        bilateral_kw = dict(
            use_bilateral=self.ebsynth_config.use_bilateral,
            sigma_spatial=self.ebsynth_config.sigma_spatial,
            sigma_color=self.ebsynth_config.sigma_color,
            n_size_step=self.ebsynth_config.n_size_step,
        )
        vote_bilateral_kw = dict(
            use_bilateral=self.ebsynth_config.use_bilateral,
            sigma_spatial=self.ebsynth_config.sigma_spatial,
            sigma_color=self.ebsynth_config.sigma_color,
        )

        # --- OPTIMIZATION: Pre-compute all patches once before the loops ---
        source_style_patches = None
        target_style_resized = None
        target_style_patches = None
        source_guide_patches = None
        target_guide_patches = None
        target_modulation_patches = None

        def _extract_patches():
            nonlocal \
                source_style_patches, \
                target_style_resized, \
                target_style_patches, \
                source_guide_patches, \
                target_guide_patches, \
                target_modulation_patches
            source_style_patches = self._extract_patches(
                style_tensor, patch_size, clear_device_cache=False
            )
            if H_s == H_t and W_s == W_t:
                target_style_resized = style_tensor
                target_style_patches = source_style_patches
            else:
                target_style_resized = resample_tensor(style_tensor, H_t, W_t)
                target_style_patches = self._extract_patches(
                    target_style_resized, patch_size, clear_device_cache=False
                )
            source_guide_patches = self._extract_patches(
                source_guide_tensor, patch_size, clear_device_cache=False
            )
            if modulation_tensor.numel() > 0:
                target_guide_patches = self._extract_patches(
                    target_guide_tensor, patch_size, clear_device_cache=False
                )
                target_modulation_patches = self._extract_patches(
                    modulation_tensor, patch_size, clear_device_cache=True
                )
            else:
                target_guide_patches = self._extract_patches(
                    target_guide_tensor, patch_size, clear_device_cache=True
                )
                target_modulation_patches = None

        self._timed_operation("patch_extraction", _extract_patches)
        # --- END OPTIMIZATION ---

        omega_map = None
        error_map = None
        omega_best = None

        def _init_omega_and_error():
            nonlocal omega_map, error_map, omega_best
            omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
            error_map = torch.full(
                (H_t, W_t), float("inf"), dtype=torch.float32, device=self.device
            )
            omega_best = (H_t * W_t) / (H_s * W_s)

        self._timed_operation("omega_initialization", _init_omega_and_error)
        clear_torch_device_cache(self.device)

        def _init_nnf():
            nonlocal \
                source_style_patches, \
                target_style_patches, \
                source_guide_patches, \
                target_guide_patches, \
                target_modulation_patches
            # ``try_patch_batch`` treats candidate/current coordinates as read-only.
            try_patch_batch(
                nnf,
                nnf,
                error_map,
                omega_map,
                source_style_patches,
                target_style_patches,
                source_guide_patches,
                target_guide_patches,
                style_weights,
                guide_weights,
                uniformity_weight,
                patch_size,
                cost_function_mode,
                omega_best,
                target_modulation_patches=target_modulation_patches,
                **bilateral_kw,
            )

        self._timed_operation("nnf_initialization", _init_nnf)

        convergence_mask = torch.full(
            (H_t, W_t), 255, dtype=torch.uint8, device=self.device
        )

        # Main PatchMatch phase (matches CUDA / CPU dispatch / Taichi / iterative
        # PyTorch): ``num_patch_match_iters`` propagation-only steps, then one
        # random-search step — not propagation and random search interleaved.
        def _run_patchmatch_iterations():
            for iteration in range(patch_match_iters):

                def _propagation():
                    nonlocal \
                        source_style_patches, \
                        target_style_patches, \
                        source_guide_patches, \
                        target_guide_patches
                    is_odd = (iteration % 2) == 1
                    propagation_step(
                        nnf,
                        error_map,
                        omega_map,
                        # MODIFIED: Pass pre-computed patches
                        source_style_patches,
                        target_style_patches,
                        source_guide_patches,
                        target_guide_patches,
                        style_weights,
                        guide_weights,
                        uniformity_weight,
                        patch_size,
                        is_odd,
                        convergence_mask,
                        cost_function_mode,
                        omega_best,
                        target_modulation_patches=target_modulation_patches,
                        **bilateral_kw,
                    )

                self._timed_operation("propagation_step", _propagation)

            def _random_search():
                nonlocal \
                    source_style_patches, \
                    target_style_patches, \
                    source_guide_patches, \
                    target_guide_patches
                random_search_step(
                    nnf,
                    error_map,
                    omega_map,
                    # MODIFIED: Pass pre-computed patches
                    source_style_patches,
                    target_style_patches,
                    source_guide_patches,
                    target_guide_patches,
                    style_weights,
                    guide_weights,
                    uniformity_weight,
                    patch_size,
                    max(H_s, W_s) // 2,
                    convergence_mask,
                    self.ebsynth_config.search_pruning_threshold,
                    cost_function_mode,
                    omega_best,
                    generator=random_generator,
                    target_modulation_patches=target_modulation_patches,
                    **bilateral_kw,
                )

            self._timed_operation("random_search_step", _random_search)

        self._timed_operation("patchmatch_iterations", _run_patchmatch_iterations)

        # Additional search-vote iterations
        def _run_search_vote_iterations():
            for _ in range(search_vote_iters):

                def _random_search():
                    nonlocal \
                        source_style_patches, \
                        target_style_patches, \
                        source_guide_patches, \
                        target_guide_patches
                    random_search_step(
                        nnf,
                        error_map,
                        omega_map,
                        # MODIFIED: Pass pre-computed patches
                        source_style_patches,
                        target_style_patches,
                        source_guide_patches,
                        target_guide_patches,
                        style_weights,
                        guide_weights,
                        uniformity_weight,
                        patch_size,
                        max(H_s, W_s) // 2,
                        convergence_mask,
                        self.ebsynth_config.search_pruning_threshold,
                        cost_function_mode,
                        omega_best,
                        generator=random_generator,
                        target_modulation_patches=target_modulation_patches,
                        **bilateral_kw,
                    )

                self._timed_operation("random_search_step", _random_search)

        self._timed_operation("search_vote_iterations", _run_search_vote_iterations)

        output_image = None

        def _perform_voting():
            nonlocal output_image
            if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
                output_image = vote_weighted(
                    style_tensor, nnf, error_map, patch_size, **vote_bilateral_kw
                )
            else:
                output_image = vote_plain(
                    style_tensor, nnf, patch_size, **vote_bilateral_kw
                )

        self._timed_operation("voting", _perform_voting)

        output_error = error_map

        del (
            source_style_patches,
            target_style_patches,
            target_style_resized,
            source_guide_patches,
            target_guide_patches,
        )
        if target_modulation_patches is not None:
            del target_modulation_patches
        clear_torch_device_cache(self.device)

        return output_image, output_error, nnf

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
        bilateral_kw = dict(
            use_bilateral=self.ebsynth_config.use_bilateral,
            sigma_spatial=self.ebsynth_config.sigma_spatial,
            sigma_color=self.ebsynth_config.sigma_color,
            n_size_step=self.ebsynth_config.n_size_step,
        )
        vote_bilateral_kw = dict(
            use_bilateral=self.ebsynth_config.use_bilateral,
            sigma_spatial=self.ebsynth_config.sigma_spatial,
            sigma_color=self.ebsynth_config.sigma_color,
        )

        # Pre-extract all source patches once, as they don't change.
        source_style_patches = None
        source_guide_patches = None
        target_guide_patches = None
        target_modulation_patches = None

        def _extract_source_patches():
            nonlocal \
                source_style_patches, \
                source_guide_patches, \
                target_guide_patches, \
                target_modulation_patches
            source_style_patches = self._extract_patches(
                style_tensor, patch_size, clear_device_cache=False
            )
            source_guide_patches = self._extract_patches(
                source_guide_tensor, patch_size, clear_device_cache=False
            )
            # --- OPTIMIZATION: Hoist target guide patch extraction out of the loop ---
            # The target guide tensor does not change during the iterative process.
            if modulation_tensor.numel() > 0:
                target_guide_patches = self._extract_patches(
                    target_guide_tensor, patch_size, clear_device_cache=False
                )
                target_modulation_patches = self._extract_patches(
                    modulation_tensor, patch_size, clear_device_cache=True
                )
            else:
                target_guide_patches = self._extract_patches(
                    target_guide_tensor, patch_size, clear_device_cache=True
                )
                target_modulation_patches = None

        self._timed_operation("patch_extraction", _extract_source_patches)

        omega_map = None
        omega_best = None

        def _init_omega_iterative():
            nonlocal omega_map, omega_best
            omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
            omega_best = (H_t * W_t) / (H_s * W_s)
            if omega_best < 1e-6:
                omega_best = 1e-6

        self._timed_operation("omega_initialization", _init_omega_iterative)
        clear_torch_device_cache(self.device)

        # Match cpu/dispatch_cpu.cpp: initial vote uses an all-zero error map (extension
        # allocates output_error with zeros). Do not run try_patch before this vote.
        target_style_temp = vote_plain(
            style_tensor, nnf, patch_size, **vote_bilateral_kw
        )

        # One buffer for the previous frame; reuse to avoid a full H×W×C clone each iteration.
        target_style_prev = torch.empty_like(target_style_temp)
        target_style_prev.copy_(target_style_temp)
        mask = torch.full((H_t, W_t), 255, dtype=torch.uint8, device=self.device)

        # Reused for every try_patch_batch that starts from an all-inf error map (read-only input).
        scratch_errors_inf = torch.full(
            (H_t, W_t), float("inf"), dtype=torch.float32, device=self.device
        )

        target_style_patches_current = None
        error_map = None
        target_style_temp = None

        def _run_iterative_refinement():
            nonlocal \
                target_style_patches_current, \
                error_map, \
                target_style_temp, \
                target_style_prev
            for iteration in range(search_vote_iters):

                def _extract_target_patches():
                    nonlocal target_style_patches_current
                    target_style_patches_current = self._extract_patches(
                        target_style_prev, patch_size
                    )

                self._timed_operation(
                    "target_patch_extraction", _extract_target_patches
                )

                def _update_nnf():
                    nonlocal \
                        error_map, \
                        scratch_errors_inf, \
                        source_style_patches, \
                        target_style_patches_current, \
                        source_guide_patches, \
                        target_guide_patches
                    error_map = try_patch_batch(
                        nnf,
                        nnf,
                        scratch_errors_inf,
                        omega_map,
                        source_style_patches,
                        target_style_patches_current,
                        source_guide_patches,
                        target_guide_patches,
                        style_weights,  # Use pre-computed target_guide_patches
                        guide_weights,
                        uniformity_weight,
                        patch_size,
                        cost_function_mode,
                        omega_best,
                        target_modulation_patches=target_modulation_patches,
                        **bilateral_kw,
                    )[1]

                self._timed_operation("nnf_update", _update_nnf)

                def _run_patchmatch_iters():
                    for pm_iter in range(patch_match_iters):

                        def _propagation():
                            nonlocal \
                                source_style_patches, \
                                target_style_patches_current, \
                                source_guide_patches, \
                                target_guide_patches
                            is_odd = (pm_iter % 2) == 1
                            propagation_step(
                                nnf,
                                error_map,
                                omega_map,
                                source_style_patches,
                                target_style_patches_current,
                                source_guide_patches,
                                target_guide_patches,
                                style_weights,
                                guide_weights,  # Use pre-computed target_guide_patches
                                uniformity_weight,
                                patch_size,
                                is_odd,
                                mask,
                                cost_function_mode,
                                omega_best,
                                target_modulation_patches=target_modulation_patches,
                                **bilateral_kw,
                            )

                        self._timed_operation("propagation_step", _propagation)

                self._timed_operation("patchmatch_iterations", _run_patchmatch_iters)

                def _final_random_search():
                    nonlocal \
                        source_style_patches, \
                        target_style_patches_current, \
                        source_guide_patches, \
                        target_guide_patches
                    random_search_step(
                        nnf,
                        error_map,
                        omega_map,
                        source_style_patches,
                        target_style_patches_current,
                        source_guide_patches,
                        target_guide_patches,
                        style_weights,
                        guide_weights,  # Use pre-computed target_guide_patches
                        uniformity_weight,
                        patch_size,
                        max(H_s, W_s) // 2,
                        mask,
                        self.ebsynth_config.search_pruning_threshold,
                        cost_function_mode,
                        omega_best,
                        generator=random_generator,
                        target_modulation_patches=target_modulation_patches,
                        **bilateral_kw,
                    )

                self._timed_operation("random_search_step", _final_random_search)

                # Free target_style_patches_current immediately
                del target_style_patches_current
                target_style_patches_current = None

                def _vote():
                    nonlocal target_style_temp
                    if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
                        target_style_temp = vote_weighted(
                            style_tensor,
                            nnf,
                            error_map,
                            patch_size,
                            **vote_bilateral_kw,
                        )
                    else:
                        target_style_temp = vote_plain(
                            style_tensor, nnf, patch_size, **vote_bilateral_kw
                        )

                self._timed_operation("voting", _vote)

                if iteration < search_vote_iters - 1 and stop_threshold > 0:

                    def _evaluate_mask():
                        nonlocal mask
                        new_mask = evaluate_mask(
                            target_style_temp, target_style_prev, stop_threshold
                        )
                        mask = dilate_mask(new_mask, patch_size)

                    self._timed_operation("mask_evaluation", _evaluate_mask)

                target_style_prev.copy_(target_style_temp)

        self._timed_operation("iterative_refinement", _run_iterative_refinement)

        if search_vote_iters == 0:
            target_style_temp = target_style_prev

        output_image = None
        output_error = None

        def _generate_final_output():
            nonlocal \
                output_image, \
                output_error, \
                scratch_errors_inf, \
                source_style_patches, \
                source_guide_patches, \
                target_guide_patches
            output_image = target_style_temp
            final_target_patches = self._extract_patches(output_image, patch_size)
            # Use pre-computed target_guide_patches here as well
            output_error = try_patch_batch(
                nnf,
                nnf,
                scratch_errors_inf,
                omega_map,
                source_style_patches,
                final_target_patches,
                source_guide_patches,
                target_guide_patches,
                style_weights,
                guide_weights,
                uniformity_weight,
                patch_size,
                cost_function_mode,
                omega_best,
                target_modulation_patches=target_modulation_patches,
                **bilateral_kw,
            )[1]
            del final_target_patches

        self._timed_operation("final_output", _generate_final_output)

        del (
            source_style_patches,
            source_guide_patches,
            target_guide_patches,
            scratch_errors_inf,
            target_style_prev,
        )
        if target_modulation_patches is not None:
            del target_modulation_patches
        clear_torch_device_cache(self.device)

        return output_image, output_error, nnf
