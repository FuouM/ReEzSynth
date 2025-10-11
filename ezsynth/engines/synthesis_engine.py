# ezsynth/engines/synthesis_engine.py
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ezsynth.torch_ops.mask_ops import dilate_mask, evaluate_mask
from ezsynth.torch_ops.patch_ops import extract_patches

from ..config import EbsynthParamsConfig, PipelineConfig
from ..torch_ops import (
    populate_omega_map,
    propagation_step,
    random_search_step,
    try_patch_batch,
    vote_plain,
    vote_weighted,
)

# Skip direct import of ebsynth_torch
# os.environ["FORCE_EBSYNTH_JIT_LOADER"] = "1"
os.environ["JIT_VERBOSE"] = "0"

# Check if we should force the JIT loader (skip direct import)
force_jit = os.getenv("FORCE_EBSYNTH_JIT_LOADER", "").lower() in ("1", "true", "yes")
if force_jit:
    if os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"):
        print("Forcing JIT loader for ebsynth_torch (direct import disabled).")

if not force_jit:
    # First, try direct import of ebsynth_torch (if installed via pip)
    try:
        import ebsynth_torch

        CUDA_EXTENSION_AVAILABLE = True
        if os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"):
            print("CUDA extension loaded successfully (direct import).")
    except ImportError:
        force_jit = True  # Fall back to JIT if direct import fails

if force_jit:
    # Try the JIT loader
    try:
        from ebsynth_torch_loader import ebsynth_torch

        CUDA_EXTENSION_AVAILABLE = ebsynth_torch is not None
        if CUDA_EXTENSION_AVAILABLE:
            if os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"):
                print("CUDA extension loaded successfully (via JIT loader).")
        else:
            print("JIT loader found but extension not available.")
    except ImportError:
        print("\nCould not find the JIT loader module.")
        ebsynth_torch = None
        CUDA_EXTENSION_AVAILABLE = False


if not CUDA_EXTENSION_AVAILABLE:
    print("\n[WARNING] PyTorch CUDA extension not available.")
    print("CUDA backend will not be available. Only PyTorch backend can be used.")
    print(
        "To enable CUDA backend, ensure a C++ compiler and CUDA Toolkit are installed.\n"
    )

# --- Constants from ebsynth.h for clarity ---
EBSYNTH_VOTEMODE_PLAIN = 0x0001
EBSYNTH_VOTEMODE_WEIGHTED = 0x0002

COST_FUNCTION_SSD = 0
COST_FUNCTION_NCC = 1


class EbsynthEngine:
    """
    A high-performance wrapper for the ebsynth library using a native
    PyTorch C++/CUDA extension. This engine manages the pyramidal synthesis
    process by calling a single-level CUDA kernel in a loop.
    """

    def __init__(
        self, ebsynth_config: EbsynthParamsConfig, pipeline_config: PipelineConfig
    ):
        """
        Initializes the EbsynthEngine.

        Args:
            ebsynth_config (EbsynthParamsConfig): Configuration for low-level ebsynth parameters.
            pipeline_config (PipelineConfig): Configuration for pipeline-level parameters.
        """
        print("Initializing Ebsynth Torch Engine...")
        self.ebsynth_config = ebsynth_config
        self.pipeline_config = pipeline_config
        self.backend = ebsynth_config.backend

        if self.backend == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA backend selected but CUDA is not available.")
            if not CUDA_EXTENSION_AVAILABLE:
                raise RuntimeError(
                    "CUDA backend selected but ebsynth_torch extension is not available."
                )
            self.device = "cuda"
            self.use_cuda_extension = True
        elif self.backend == "torch":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.use_cuda_extension = False
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.rand_states = None
        self.vote_mode_map = {
            "weighted": EBSYNTH_VOTEMODE_WEIGHTED,
            "plain": EBSYNTH_VOTEMODE_PLAIN,
        }
        self.cost_function_map = {
            "ssd": COST_FUNCTION_SSD,
            "ncc": COST_FUNCTION_NCC,
        }
        print(
            f"Ebsynth Engine initialized with backend: '{self.backend}' on device: '{self.device}'"
        )

    def _resample_tensor(
        self, tensor: torch.Tensor, new_h: int, new_w: int, mode: str = "bilinear"
    ) -> torch.Tensor:
        """Helper to resample a tensor using torch.nn.functional.interpolate."""
        if tensor.shape[0] == new_h and tensor.shape[1] == new_w:
            return tensor

        is_uint8 = tensor.dtype == torch.uint8

        tensor_float = tensor.permute(2, 0, 1).unsqueeze(0).float()

        resampled_float = F.interpolate(
            tensor_float, size=(new_h, new_w), mode=mode, align_corners=False
        )

        resampled = resampled_float.squeeze(0).permute(1, 2, 0)

        if is_uint8:
            return resampled.clamp(0, 255).to(torch.uint8).contiguous()

        return resampled.contiguous()

    def _init_nnf(self, target_h, target_w, source_h, source_w, patch_size):
        """Initializes a random NNF on the GPU."""
        r = patch_size // 2
        rand_x = torch.randint(
            r,
            source_w - r,
            (target_h, target_w, 1),
            device=self.device,
            dtype=torch.int32,
        )
        rand_y = torch.randint(
            r,
            source_h - r,
            (target_h, target_w, 1),
            device=self.device,
            dtype=torch.int32,
        )
        return torch.cat([rand_x, rand_y], dim=2).contiguous()

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

    def run(
        self,
        style_img: np.ndarray,
        guides: List[Tuple[np.ndarray, np.ndarray, float]],
        modulation_map: Optional[np.ndarray] = None,
        initial_nnf: Optional[np.ndarray] = None,
        output_nnf: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Runs the full pyramidal synthesis process.
        """
        # --- 1. Prepare Tensors and move to the correct device ---
        style_tensor = torch.from_numpy(style_img).to(self.device)
        guides_source_np = [g[0] for g in guides]
        guides_target_np = [g[1] for g in guides]
        guide_weights_py = [g[2] for g in guides]

        source_guide_cat = torch.from_numpy(
            np.concatenate(guides_source_np, axis=2)
        ).to(self.device)
        target_guide_cat = torch.from_numpy(
            np.concatenate(guides_target_np, axis=2)
        ).to(self.device)

        modulation_tensor = torch.empty(0, device=self.device, dtype=torch.uint8)
        if modulation_map is not None:
            modulation_tensor = torch.from_numpy(modulation_map).to(self.device)

        sh, sw, sc = style_tensor.shape
        th, tw, _ = target_guide_cat.shape

        if (
            self.rand_states is None
            or self.rand_states.numel() * self.rand_states.element_size() < th * tw * 48
        ):
            self.rand_states = torch.empty(
                th * tw * 48, dtype=torch.uint8, device=self.device
            )
            ebsynth_torch.init_rand_states(self.rand_states)

        # --- 2. Determine Pyramid Levels ---
        num_pyramid_levels = self.pipeline_config.pyramid_levels
        max_levels = 0
        min_dim_start = min(sh, sw, th, tw)
        for level in range(32, -1, -1):
            if (min_dim_start * (2.0**-level)) >= (
                2 * self.ebsynth_config.patch_size + 1
            ):
                max_levels = level + 1
                break

        num_pyramid_levels = min(num_pyramid_levels, max_levels)

        # --- 3. Prepare Pyramid Data ---
        p_style, p_source_guide, p_target_guide, p_modulation = [], [], [], []
        for i in range(num_pyramid_levels):
            scale = 2.0 ** -(num_pyramid_levels - 1 - i)
            p_sh, p_sw = max(1, int(sh * scale)), max(1, int(sw * scale))
            p_th, p_tw = max(1, int(th * scale)), max(1, int(tw * scale))

            p_style.append(self._resample_tensor(style_tensor, p_sh, p_sw))
            p_source_guide.append(self._resample_tensor(source_guide_cat, p_sh, p_sw))
            p_target_guide.append(self._resample_tensor(target_guide_cat, p_th, p_tw))
            if modulation_map is not None:
                p_modulation.append(
                    self._resample_tensor(modulation_tensor, p_th, p_tw)
                )
            else:
                p_modulation.append(
                    torch.empty(0, device=self.device, dtype=torch.uint8)
                )

        # --- 4. Main Pyramid Loop ---
        nnf = None
        output_image = None
        output_error = None
        vote_mode = self.vote_mode_map[self.ebsynth_config.vote_mode]
        cost_function_mode = self.cost_function_map[self.ebsynth_config.cost_function]

        for level in range(num_pyramid_levels):
            p_style_level = p_style[level]
            p_source_guide_level = p_source_guide[level]
            p_target_guide_level = p_target_guide[level]
            p_modulation_level = p_modulation[level]

            p_sh, p_sw, _ = p_style_level.shape
            p_th, p_tw, _ = p_target_guide_level.shape

            if level == 0:
                if initial_nnf is not None:
                    # initial_nnf is full-res, downsample it for the first, coarsest level
                    initial_nnf_tensor = (
                        torch.from_numpy(initial_nnf).to(self.device).contiguous()
                    )

                    scale_h = p_th / th
                    scale_w = p_tw / tw

                    nnf_float = initial_nnf_tensor.permute(2, 0, 1).unsqueeze(0).float()
                    resampled_nnf = F.interpolate(
                        nnf_float,
                        size=(p_th, p_tw),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # Scale the coordinate values within the NNF. NNF shape is (H, W, 2) with (x, y) coords.
                    resampled_nnf[:, 0, :, :] *= scale_w
                    resampled_nnf[:, 1, :, :] *= scale_h

                    nnf = (
                        resampled_nnf.squeeze(0)
                        .permute(1, 2, 0)
                        .to(torch.int32)
                        .contiguous()
                    )
                else:
                    # No initial NNF, so randomly initialize for the first level
                    nnf = self._init_nnf(
                        p_th, p_tw, p_sh, p_sw, self.ebsynth_config.patch_size
                    )
            else:
                # Upsample NNF from previous coarser level
                nnf_float = nnf.permute(2, 0, 1).unsqueeze(0).float() * 2.0
                upscaled_nnf = F.interpolate(
                    nnf_float, size=(p_th, p_tw), mode="bilinear", align_corners=False
                )
                nnf = (
                    upscaled_nnf.squeeze(0)
                    .permute(1, 2, 0)
                    .to(torch.int32)
                    .contiguous()
                )

            nnf[..., 0].clamp_(min=0, max=p_sw - 1)
            nnf[..., 1].clamp_(min=0, max=p_sh - 1)

            num_style_channels = p_style_level.shape[2]
            style_weights = torch.tensor(
                [1.0 / num_style_channels] * num_style_channels,
                dtype=torch.float32,
                device=self.device,
            )

            guide_weights_list = []
            for gs_np, weight in zip(guides_source_np, guide_weights_py):
                num_guide_channels = gs_np.shape[2]
                guide_weights_list.extend(
                    [weight / num_guide_channels] * num_guide_channels
                )
            guide_weights = torch.tensor(
                guide_weights_list, dtype=torch.float32, device=self.device
            )

            if self.use_cuda_extension:
                output_image, output_error, nnf = ebsynth_torch.run_level(
                    p_style_level,
                    p_source_guide_level,
                    p_target_guide_level,
                    p_modulation_level,
                    nnf,
                    style_weights,
                    guide_weights,
                    self.ebsynth_config.uniformity,
                    self.ebsynth_config.patch_size,
                    vote_mode,
                    self.ebsynth_config.search_vote_iters,
                    self.ebsynth_config.patch_match_iters,
                    self.ebsynth_config.stop_threshold,
                    self.rand_states,
                    self.ebsynth_config.search_pruning_threshold,
                    cost_function_mode,  # Pass new parameter
                )
            else:
                if self.pipeline_config.use_residual_transfer:
                    output_image, output_error, nnf = self._run_level_pytorch_iterative(
                        p_style_level,
                        p_source_guide_level,
                        p_target_guide_level,
                        p_modulation_level,
                        nnf,
                        style_weights,
                        guide_weights,
                        self.ebsynth_config.uniformity,
                        self.ebsynth_config.patch_size,
                        vote_mode,
                        self.ebsynth_config.search_vote_iters,
                        self.ebsynth_config.patch_match_iters,
                        self.ebsynth_config.stop_threshold,
                        cost_function_mode,
                    )
                else:
                    output_image, output_error, nnf = self._run_level_pytorch(
                        p_style_level,
                        p_source_guide_level,
                        p_target_guide_level,
                        p_modulation_level,
                        nnf,
                        style_weights,
                        guide_weights,
                        self.ebsynth_config.uniformity,
                        self.ebsynth_config.patch_size,
                        vote_mode,
                        self.ebsynth_config.search_vote_iters,
                        self.ebsynth_config.patch_match_iters,
                        self.ebsynth_config.stop_threshold,
                        cost_function_mode,
                    )

        # --- 5. Optional Extra Pass 3x3 ---
        if self.ebsynth_config.extra_pass_3x3:
            print("  - Performing final 3x3 pass...")

            guide_weights_list = []
            for gs_np, weight in zip(guides_source_np, guide_weights_py):
                num_guide_channels = gs_np.shape[2]
                guide_weights_list.extend(
                    [weight / num_guide_channels] * num_guide_channels
                )
            guide_weights = torch.tensor(
                guide_weights_list, dtype=torch.float32, device=self.device
            )

            style_weights = torch.tensor(
                [1.0 / sc] * sc, dtype=torch.float32, device=self.device
            )

            if self.use_cuda_extension:
                output_image, output_error, nnf = ebsynth_torch.run_level(
                    style_tensor,
                    source_guide_cat,
                    target_guide_cat,
                    modulation_tensor,
                    nnf,
                    style_weights,
                    guide_weights,
                    0.0,
                    3,
                    vote_mode,
                    self.ebsynth_config.search_vote_iters,
                    self.ebsynth_config.patch_match_iters,
                    self.ebsynth_config.stop_threshold,
                    self.rand_states,
                    self.ebsynth_config.search_pruning_threshold,
                    cost_function_mode,  # Pass new parameter
                )
            else:
                output_image, output_error, nnf = self._run_level_pytorch(
                    style_tensor,
                    source_guide_cat,
                    target_guide_cat,
                    modulation_tensor,
                    nnf,
                    style_weights,
                    guide_weights,
                    0.0,  # uniformity_weight = 0.0 for 3x3 pass
                    3,  # patch_size = 3
                    vote_mode,
                    self.ebsynth_config.search_vote_iters,
                    self.ebsynth_config.patch_match_iters,
                    self.ebsynth_config.stop_threshold,
                    cost_function_mode,
                )

        # --- 6. Convert final results back to NumPy arrays ---
        stylized_image_np = output_image.cpu().numpy()
        error_map_np = output_error.cpu().numpy()

        if output_nnf:
            nnf_np = nnf.cpu().numpy()
            return stylized_image_np, error_map_np, nnf_np

        return stylized_image_np, error_map_np
