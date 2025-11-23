# ezsynth/engines/backends/pytorch_backend.py
from typing import Optional, Tuple

import torch

from ...config import EbsynthParamsConfig, PipelineConfig
from ...torch_ops import (
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

from ...consts import EBSYNTH_VOTEMODE_PLAIN, EBSYNTH_VOTEMODE_WEIGHTED


class PyTorchBackend(BaseSynthesisBackend):
    """
    PyTorch backend with CPU/GPU support using torch_ops.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        super().__init__(ebsynth_config, pipeline_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a single level using PyTorch operations.
        """
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
        source_style_patches = extract_patches(style_tensor, patch_size)
        target_style_resized = self._resample_tensor(style_tensor, H_t, W_t)
        target_style_patches = extract_patches(target_style_resized, patch_size)
        source_guide_patches = extract_patches(source_guide_tensor, patch_size)
        target_guide_patches = extract_patches(target_guide_tensor, patch_size)
        # --- END OPTIMIZATION ---

        omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
        error_map = torch.full(
            (H_t, W_t), float("inf"), dtype=torch.float32, device=self.device
        )
        omega_best = (H_t * W_t) / (H_s * W_s)

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

        convergence_mask = torch.full(
            (H_t, W_t), 255, dtype=torch.uint8, device=self.device
        )

        # Main PatchMatch iterations
        for iteration in range(patch_match_iters):
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

        # Additional search-vote iterations
        for _ in range(search_vote_iters):
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

        if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
            output_image = vote_weighted(style_tensor, nnf, error_map, patch_size)
        else:
            output_image = vote_plain(style_tensor, nnf, patch_size)

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
        source_style_patches = extract_patches(style_tensor, patch_size)
        source_guide_patches = extract_patches(source_guide_tensor, patch_size)
        # --- OPTIMIZATION: Hoist target guide patch extraction out of the loop ---
        # The target guide tensor does not change during the iterative process.
        target_guide_patches = extract_patches(target_guide_tensor, patch_size)

        omega_map = populate_omega_map(nnf, (H_s, W_s), patch_size)
        omega_best = (H_t * W_t) / (H_s * W_s)
        if omega_best < 1e-6:
            omega_best = 1e-6

        initial_target_patches = extract_patches(
            self._resample_tensor(style_tensor, H_t, W_t), patch_size
        )
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

        if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
            target_style_temp = vote_weighted(
                style_tensor, nnf, initial_error_map, patch_size
            )
        else:
            target_style_temp = vote_plain(style_tensor, nnf, patch_size)

        target_style_prev = target_style_temp.clone()
        mask = torch.full((H_t, W_t), 255, dtype=torch.uint8, device=self.device)

        for iteration in range(search_vote_iters):
            target_style_patches_current = extract_patches(
                target_style_prev, patch_size
            )
            # No longer need to extract target guide patches here.

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

            for pm_iter in range(patch_match_iters):
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
            del target_style_patches_current

            if vote_mode == EBSYNTH_VOTEMODE_WEIGHTED:
                target_style_temp = vote_weighted(
                    style_tensor, nnf, error_map, patch_size
                )
            else:
                target_style_temp = vote_plain(style_tensor, nnf, patch_size)

            if iteration < search_vote_iters - 1 and stop_threshold > 0:
                new_mask = evaluate_mask(
                    target_style_temp, target_style_prev, stop_threshold
                )
                mask = dilate_mask(new_mask, patch_size)

            target_style_prev = target_style_temp.clone()

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

        return output_image, output_error, nnf
