# ezsynth/engines/synthesis_engine.py
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EbsynthParamsConfig, PipelineConfig
from ..consts import (
    COST_FUNCTION_NCC,
    COST_FUNCTION_SSD,
    EBSYNTH_VOTEMODE_PLAIN,
    EBSYNTH_VOTEMODE_WEIGHTED,
    ebsynth_torch,
)
from ..torch_ops import SynthesisTimer
from .backends import CudaBackend, PyTorchBackend


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
        self.backend_type = ebsynth_config.backend

        # Create the appropriate backend
        if self.backend_type == "cuda":
            self.backend = CudaBackend(ebsynth_config, pipeline_config)
        elif self.backend_type == "torch":
            self.backend = PyTorchBackend(ebsynth_config, pipeline_config)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        self.device = self.backend.device
        self.rand_states = None
        self.timer = SynthesisTimer()
        self.benchmark_enabled = False
        self.vote_mode_map = {
            "weighted": EBSYNTH_VOTEMODE_WEIGHTED,
            "plain": EBSYNTH_VOTEMODE_PLAIN,
        }
        self.cost_function_map = {
            "ssd": COST_FUNCTION_SSD,
            "ncc": COST_FUNCTION_NCC,
        }
        print(
            f"Ebsynth Engine initialized with backend: '{self.backend_type}' on device: '{self.device}'"
        )

    def _timed_operation(self, operation_name: str, operation_func):
        """Execute an operation with optional timing."""
        if self.benchmark_enabled:
            with self.timer.time_operation(operation_name):
                return operation_func()
        else:
            return operation_func()

    def run(
        self,
        style_img: np.ndarray,
        guides: List[Tuple[np.ndarray, np.ndarray, float]],
        modulation_map: Optional[np.ndarray] = None,
        initial_nnf: Optional[np.ndarray] = None,
        output_nnf: bool = False,
        benchmark: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Runs the full pyramidal synthesis process.
        """
        if benchmark:
            self.benchmark_enabled = True
            self.timer.reset()
            # Also enable backend benchmarking
            if hasattr(self.backend, "enable_benchmarking"):
                self.backend.enable_benchmarking(True)

        # Declare variables for use in nested functions
        style_tensor = None
        guides_source_np = None
        guides_target_np = None
        guide_weights_py = None
        source_guide_cat = None
        target_guide_cat = None
        modulation_tensor = None

        # --- 1. Prepare Tensors and move to the correct device ---
        def _prepare_tensors():
            nonlocal style_tensor, guides_source_np, guides_target_np, guide_weights_py
            nonlocal source_guide_cat, target_guide_cat, modulation_tensor

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

        self._timed_operation("tensor_preparation", _prepare_tensors)

        sh, sw, sc = style_tensor.shape
        th, tw, _ = target_guide_cat.shape

        if (
            self.rand_states is None
            or self.rand_states.numel() * self.rand_states.element_size() < th * tw * 48
        ):
            self.rand_states = torch.empty(
                th * tw * 48, dtype=torch.uint8, device=self.device
            )
            # Only initialize rand_states for CUDA backend
            if self.backend_type == "cuda":
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

        # Declare pyramid variables for use in nested functions
        p_style = None
        p_source_guide = None
        p_target_guide = None
        p_modulation = None

        # --- 3. Prepare Pyramid Data ---
        def _prepare_pyramid():
            nonlocal p_style, p_source_guide, p_target_guide, p_modulation
            p_style, p_source_guide, p_target_guide, p_modulation = [], [], [], []
            for i in range(num_pyramid_levels):
                scale = 2.0 ** -(num_pyramid_levels - 1 - i)
                p_sh, p_sw = max(1, int(sh * scale)), max(1, int(sw * scale))
                p_th, p_tw = max(1, int(th * scale)), max(1, int(tw * scale))

                p_style.append(self.backend._resample_tensor(style_tensor, p_sh, p_sw))
                p_source_guide.append(
                    self.backend._resample_tensor(source_guide_cat, p_sh, p_sw)
                )
                p_target_guide.append(
                    self.backend._resample_tensor(target_guide_cat, p_th, p_tw)
                )
                if modulation_map is not None:
                    p_modulation.append(
                        self.backend._resample_tensor(modulation_tensor, p_th, p_tw)
                    )
                else:
                    p_modulation.append(
                        torch.empty(0, device=self.device, dtype=torch.uint8)
                    )

        self._timed_operation("pyramid_preparation", _prepare_pyramid)

        # Declare pyramid loop variables for use in nested functions
        nnf = None
        output_image = None
        output_error = None
        vote_mode = None
        cost_function_mode = None

        # --- 4. Main Pyramid Loop ---
        def _run_pyramid_loop():
            nonlocal nnf, output_image, output_error
            nnf = None
            output_image = None
            output_error = None
            vote_mode = self.vote_mode_map[self.ebsynth_config.vote_mode]
            cost_function_mode = self.cost_function_map[
                self.ebsynth_config.cost_function
            ]

            for level in range(num_pyramid_levels):

                def _process_pyramid_level():
                    nonlocal nnf, output_image, output_error
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
                                torch.from_numpy(initial_nnf)
                                .to(self.device)
                                .contiguous()
                            )

                            scale_h = p_th / th
                            scale_w = p_tw / tw

                            nnf_float = (
                                initial_nnf_tensor.permute(2, 0, 1).unsqueeze(0).float()
                            )
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
                            nnf = self.backend._init_nnf(
                                p_th, p_tw, p_sh, p_sw, self.ebsynth_config.patch_size
                            )
                    else:
                        # Upsample NNF from previous coarser level
                        nnf_float = nnf.permute(2, 0, 1).unsqueeze(0).float() * 2.0
                        upscaled_nnf = F.interpolate(
                            nnf_float,
                            size=(p_th, p_tw),
                            mode="bilinear",
                            align_corners=False,
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

                    output_image, output_error, nnf = self.backend.run_level(
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
                        cost_function_mode,
                        benchmark,
                    )

                self._timed_operation(f"pyramid_level_{level}", _process_pyramid_level)

        self._timed_operation("pyramid_processing", _run_pyramid_loop)

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

            output_image, output_error, nnf = self.backend.run_level(
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
                self.rand_states,
                cost_function_mode,
                benchmark,
            )

        # Declare result variables for use in nested functions
        stylized_image_np = None
        error_map_np = None

        # --- 6. Convert final results back to NumPy arrays ---
        def _convert_results():
            nonlocal stylized_image_np, error_map_np
            stylized_image_np = output_image.cpu().numpy()
            error_map_np = output_error.cpu().numpy()

        self._timed_operation("result_conversion", _convert_results)

        # --- 7. Print benchmark summary if enabled ---
        if benchmark:
            print("\n" + "=" * 60)
            print("SYNTHESIS ENGINE TIMING SUMMARY")
            print("=" * 60)
            self.timer.print_summary("Synthesis Engine Operations")

            if hasattr(self.backend, "timer"):
                print("\n" + "=" * 60)
                backend_name = "CUDA" if self.backend_type == "cuda" else "PyTorch"
                print(f"{backend_name.upper()} BACKEND TIMING SUMMARY")
                print("=" * 60)
                self.backend.timer.print_summary(f"{backend_name} Backend Operations")

        if output_nnf:
            nnf_np = nnf.cpu().numpy()
            return stylized_image_np, error_map_np, nnf_np

        return stylized_image_np, error_map_np
