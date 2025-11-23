# ezsynth/engines/backends/pytorch_backend.py
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...consts import EBSYNTH_VOTEMODE_PLAIN, EBSYNTH_VOTEMODE_WEIGHTED
from ...torch_ops import (
    SynthesisTimer,
    populate_omega_map,
    propagation_step,
    random_search_step,
    try_patch_batch,
    vote_plain,
    vote_weighted,
)
from ...torch_ops.mask_ops import dilate_mask, evaluate_mask
from ...torch_ops.patch_ops import extract_patches
from .base import BaseSynthesisBackend


class PyTorchBackend(BaseSynthesisBackend):
    """
    PyTorch backend with CPU/GPU support using torch_ops.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        super().__init__(ebsynth_config, pipeline_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    ):
        """
        PyTorch implementation of the synthesis level using torch_ops.
        """
        H_s, W_s, C_s = style_tensor.shape
        H_t, W_t, C_g = target_guide_tensor.shape

        # --- OPTIMIZATION: Pre-compute all patches once before the loops ---
        source_style_patches = None
        target_style_resized = None
        target_style_patches = None
        source_guide_patches = None
        target_guide_patches = None

        def _extract_patches():
            nonlocal \
                source_style_patches, \
                target_style_resized, \
                target_style_patches, \
                source_guide_patches, \
                target_guide_patches
            source_style_patches = extract_patches(style_tensor, patch_size)
            target_style_resized = self._resample_tensor(style_tensor, H_t, W_t)
            target_style_patches = extract_patches(target_style_resized, patch_size)
            source_guide_patches = extract_patches(source_guide_tensor, patch_size)
            target_guide_patches = extract_patches(target_guide_tensor, patch_size)

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

        def _init_nnf():
            # Initialize with current NNF
            try_patch_batch(
                nnf.clone(),
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
                cost_function_mode,
                omega_best,
            )

        self._timed_operation("nnf_initialization", _init_nnf)

        convergence_mask = torch.full(
            (H_t, W_t), 255, dtype=torch.uint8, device=self.device
        )

        # Main PatchMatch iterations
        def _run_patchmatch_iterations():
            for iteration in range(patch_match_iters):

                def _propagation():
                    is_odd = (iteration % 2) == 0
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
                    )

                self._timed_operation("propagation_step", _propagation)

                def _random_search():
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
                    )

                self._timed_operation("random_search_step", _random_search)

        self._timed_operation("patchmatch_iterations", _run_patchmatch_iterations)

        # Additional search-vote iterations
        def _run_search_vote_iterations():
            for _ in range(search_vote_iters):

                def _random_search():
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
                    )

                self._timed_operation("random_search_step", _random_search)

        self._timed_operation("search_vote_iterations", _run_search_vote_iterations)

        output_image = None

        def _perform_voting():
            nonlocal output_image
            if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
                output_image = vote_weighted(style_tensor, nnf, error_map, patch_size)
            else:
                output_image = vote_plain(style_tensor, nnf, patch_size)

        self._timed_operation("voting", _perform_voting)

        output_error = error_map
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
    ):
        """
        New, fast algorithm matching CUDA. Iteratively refines the NNF and the
        synthesized image together in a loop.
        """
        H_s, W_s, _ = style_tensor.shape
        H_t, W_t, _ = target_guide_tensor.shape

        # Pre-extract all source patches once, as they don't change.
        source_style_patches = None
        source_guide_patches = None
        target_guide_patches = None

        def _extract_source_patches():
            nonlocal source_style_patches, source_guide_patches, target_guide_patches
            source_style_patches = extract_patches(style_tensor, patch_size)
            source_guide_patches = extract_patches(source_guide_tensor, patch_size)
            # --- OPTIMIZATION: Hoist target guide patch extraction out of the loop ---
            # The target guide tensor does not change during the iterative process.
            target_guide_patches = extract_patches(target_guide_tensor, patch_size)

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

        initial_target_patches = None

        def _extract_initial_target_patches():
            nonlocal initial_target_patches
            initial_target_patches = extract_patches(
                self._resample_tensor(style_tensor, H_t, W_t), patch_size
            )

        self._timed_operation("initial_target_patches", _extract_initial_target_patches)

        initial_error_map = None

        def _init_nnf_iterative():
            nonlocal initial_error_map, initial_target_patches
            initial_error_map = try_patch_batch(
                nnf,
                nnf,
                torch.full((H_t, W_t), float("inf"), device=self.device),
                omega_map,
                source_style_patches,
                initial_target_patches,
                source_guide_patches,
                target_guide_patches,
                style_weights,
                guide_weights,  # Use pre-computed target_guide_patches
                uniformity_weight,
                patch_size,
                cost_function_mode,
                omega_best,
            )[1]
            del initial_target_patches

        self._timed_operation("nnf_initialization", _init_nnf_iterative)

        if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
            target_style_temp = vote_weighted(
                style_tensor, nnf, initial_error_map, patch_size
            )
        else:
            target_style_temp = vote_plain(style_tensor, nnf, patch_size)

        target_style_prev = target_style_temp.clone()
        mask = torch.full((H_t, W_t), 255, dtype=torch.uint8, device=self.device)

        target_style_patches_current = None
        error_map = initial_error_map
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
                    target_style_patches_current = extract_patches(
                        target_style_prev, patch_size
                    )

                self._timed_operation(
                    "target_patch_extraction", _extract_target_patches
                )

                def _update_nnf():
                    nonlocal error_map
                    error_map = try_patch_batch(
                        nnf,
                        nnf,
                        torch.full_like(omega_map, float("inf"), dtype=torch.float32),
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
                    )[1]

                self._timed_operation("nnf_update", _update_nnf)

                def _run_patchmatch_iters():
                    for pm_iter in range(patch_match_iters):

                        def _propagation():
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
                            )

                        self._timed_operation("propagation_step", _propagation)

                self._timed_operation("patchmatch_iterations", _run_patchmatch_iters)

                def _final_random_search():
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
                    )

                self._timed_operation("random_search_step", _final_random_search)
                del target_style_patches_current
                target_style_patches_current = None

                def _vote():
                    nonlocal target_style_temp
                    if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
                        target_style_temp = vote_weighted(
                            style_tensor, nnf, error_map, patch_size
                        )
                    else:
                        target_style_temp = vote_plain(style_tensor, nnf, patch_size)

                self._timed_operation("voting", _vote)

                if iteration < search_vote_iters - 1 and stop_threshold > 0:

                    def _evaluate_mask():
                        nonlocal mask
                        new_mask = evaluate_mask(
                            target_style_temp, target_style_prev, stop_threshold
                        )
                        mask = dilate_mask(new_mask, patch_size)

                    self._timed_operation("mask_evaluation", _evaluate_mask)

                target_style_prev = target_style_temp.clone()

        self._timed_operation("iterative_refinement", _run_iterative_refinement)

        output_image = None
        output_error = None

        def _generate_final_output():
            nonlocal output_image, output_error
            output_image = target_style_temp
            final_target_patches = extract_patches(output_image, patch_size)
            # Use pre-computed target_guide_patches here as well
            output_error = try_patch_batch(
                nnf,
                nnf,
                torch.full_like(omega_map, float("inf"), dtype=torch.float32),
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
            )[1]

        self._timed_operation("final_output", _generate_final_output)

        return output_image, output_error, nnf
