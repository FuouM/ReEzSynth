# ezsynth/engines/synthesis_engine.py
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EbsynthParamsConfig, PipelineConfig

# --- Import our newly compiled PyTorch C++ extension ---
try:
    import ebsynth_torch
except ImportError:
    print("\n[FATAL ERROR] PyTorch extension 'ebsynth_torch' not found.")
    print("This is a required component for the synthesis engine.")
    print(
        "Please build it by running 'pip install .' in the project's root directory.\n"
    )
    raise

# --- Constants from ebsynth.h for clarity ---
EBSYNTH_VOTEMODE_PLAIN = 0x0001
EBSYNTH_VOTEMODE_WEIGHTED = 0x0002


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError("EbsynthEngine requires a CUDA-enabled device.")

        self.rand_states = None
        self.vote_mode_map = {
            "weighted": EBSYNTH_VOTEMODE_WEIGHTED,
            "plain": EBSYNTH_VOTEMODE_PLAIN,
        }
        print(f"Ebsynth Engine initialized on device: '{self.device}'")

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

    def run(
        self,
        style_img: np.ndarray,
        guides: List[Tuple[np.ndarray, np.ndarray, float]],
        modulation_map: Optional[np.ndarray] = None,
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

        for level in range(num_pyramid_levels):
            p_style_level = p_style[level]
            p_source_guide_level = p_source_guide[level]
            p_target_guide_level = p_target_guide[level]
            p_modulation_level = p_modulation[level]

            p_sh, p_sw, _ = p_style_level.shape
            p_th, p_tw, _ = p_target_guide_level.shape

            if nnf is None:
                nnf = self._init_nnf(
                    p_th, p_tw, p_sh, p_sw, self.ebsynth_config.patch_size
                )
            else:
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
            )

        # --- 6. Convert final results back to NumPy arrays ---
        stylized_image_np = output_image.cpu().numpy()
        error_map_np = output_error.cpu().numpy()

        if output_nnf:
            nnf_np = nnf.cpu().numpy()
            return stylized_image_np, error_map_np, nnf_np

        return stylized_image_np, error_map_np
