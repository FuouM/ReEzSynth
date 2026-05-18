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
from ..guide import GuideObject
from ..torch_ops import SynthesisTimer
from .backends import CudaBackend, PyTorchBackend, TaichiBackend
from .backends.common import random_init_nnf, resample_tensor

EngineRunResult = Union[
    Tuple[np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray], np.ndarray],
]


class PreparedSynthesisContext:
    """Reusable per-sequence synthesis inputs for a stable style/source-guide set."""

    def __init__(
        self,
        engine: "EbsynthEngine",
        style_img: np.ndarray,
        guides: List[GuideObject],
    ) -> None:
        _validate_image_array(style_img, "style_img")
        _validate_guides(guides)

        self.engine = engine
        self.device_type = torch.device(engine.device).type
        self.guide_channel_counts = [guide.keyframe.shape[2] for guide in guides]
        self.guide_channel_slices = _channel_slices(self.guide_channel_counts)
        self.guide_weights_py = [guide.weight for guide in guides]
        self.th, self.tw = guides[0].target.shape[:2]
        self.total_guide_channels = sum(self.guide_channel_counts)

        self.style_tensor = torch.from_numpy(style_img).to(engine.device).contiguous()
        source_guide_cat_np = np.concatenate([g.keyframe for g in guides], axis=2)
        self.source_guide_cat = (
            torch.from_numpy(source_guide_cat_np).to(engine.device).contiguous()
        )

        self.sh, self.sw, self.sc = self.style_tensor.shape
        self.num_pyramid_levels = engine._determine_num_pyramid_levels(
            self.sh, self.sw, self.th, self.tw
        )
        self.pyramid_shapes = self._build_pyramid_shapes()
        self.style_weights = _channel_weights(self.sc, 1.0, engine.device)
        self.guide_weights = _guide_weights(
            [g.keyframe for g in guides], self.guide_weights_py, engine.device
        )
        self.empty_modulation = torch.empty(0, device=engine.device, dtype=torch.uint8)
        self._target_guide_cat_np = np.empty(
            (self.th, self.tw, self.total_guide_channels), dtype=np.uint8
        )
        self._target_guide_cat_tensor: Optional[torch.Tensor] = None
        if self.device_type != "cpu":
            self._target_guide_cat_tensor = torch.empty(
                (self.th, self.tw, self.total_guide_channels),
                dtype=torch.uint8,
                device=engine.device,
            )
        self._modulation_tensor: Optional[torch.Tensor] = None
        self._modulation_shape: Optional[Tuple[int, int, int]] = None
        self._initial_nnf_tensor: Optional[torch.Tensor] = None
        self._initial_nnf_shape: Optional[Tuple[int, int, int]] = None

        self.p_style: List[torch.Tensor] = []
        self.p_source_guide: List[torch.Tensor] = []

        def _prepare_source_pyramid() -> None:
            for p_sh, p_sw, _, _ in self.pyramid_shapes:
                self.p_style.append(resample_tensor(self.style_tensor, p_sh, p_sw))
                self.p_source_guide.append(
                    resample_tensor(self.source_guide_cat, p_sh, p_sw)
                )

        engine._timed_operation("source_pyramid_preparation", _prepare_source_pyramid)

    def run_frame(
        self,
        guides: List[GuideObject],
        modulation_map: Optional[np.ndarray] = None,
        initial_nnf: Optional[np.ndarray] = None,
        return_error: bool = True,
        output_nnf: bool = False,
    ) -> EngineRunResult:
        self._validate_targets(guides)
        if modulation_map is not None:
            _validate_image_array(modulation_map, "modulation_map")

        engine = self.engine
        engine._ensure_rand_states(self.th, self.tw)

        target_guide_cat = self._targets_to_tensor(guides)

        modulation_tensor = self.empty_modulation
        if modulation_map is not None:
            modulation_tensor = self._modulation_map_tensor(modulation_map)

        empty_pyramid = [self.empty_modulation] * self.num_pyramid_levels
        p_target_guide: List[torch.Tensor] = empty_pyramid.copy()
        p_modulation: List[torch.Tensor] = empty_pyramid.copy()

        def _prepare_target_pyramid() -> None:
            for i, (_, _, p_th, p_tw) in enumerate(self.pyramid_shapes):
                p_target_guide[i] = resample_tensor(target_guide_cat, p_th, p_tw)
                if modulation_map is not None:
                    p_modulation[i] = resample_tensor(modulation_tensor, p_th, p_tw)
                else:
                    p_modulation[i] = self.empty_modulation

        engine._timed_operation("target_pyramid_preparation", _prepare_target_pyramid)

        output_image, output_error_tensor, nnf = engine._run_prepared_context(
            context=self,
            target_guide_cat=target_guide_cat,
            modulation_tensor=modulation_tensor,
            p_target_guide=p_target_guide,
            p_modulation=p_modulation,
            initial_nnf=initial_nnf,
        )

        stylized_image_np = output_image.cpu().numpy()
        error_map_np = output_error_tensor.cpu().numpy() if return_error else None
        if output_nnf:
            return stylized_image_np, error_map_np, nnf.cpu().numpy()
        return stylized_image_np, error_map_np

    def _targets_to_tensor(self, guides: List[GuideObject]) -> torch.Tensor:
        for guide, channel_slice in zip(guides, self.guide_channel_slices):
            self._target_guide_cat_np[..., channel_slice] = guide.target

        target_guide_cpu = torch.from_numpy(self._target_guide_cat_np)
        if self.device_type == "cpu":
            return target_guide_cpu
        assert self._target_guide_cat_tensor is not None
        self._target_guide_cat_tensor.copy_(target_guide_cpu, non_blocking=True)
        return self._target_guide_cat_tensor

    def _modulation_map_tensor(self, modulation_map: np.ndarray) -> torch.Tensor:
        modulation_np = _contiguous_array(modulation_map)
        if self.device_type == "cpu":
            return torch.from_numpy(modulation_np)

        if (
            self._modulation_tensor is None
            or self._modulation_shape != modulation_np.shape
        ):
            self._modulation_shape = modulation_np.shape
            self._modulation_tensor = torch.empty(
                modulation_np.shape,
                dtype=torch.uint8,
                device=self.engine.device,
            )

        self._modulation_tensor.copy_(
            torch.from_numpy(modulation_np),
            non_blocking=True,
        )
        return self._modulation_tensor

    def initial_nnf_tensor(self, initial_nnf: np.ndarray) -> torch.Tensor:
        initial_nnf_np = _contiguous_array(initial_nnf)
        if self.device_type == "cpu":
            return torch.from_numpy(initial_nnf_np)

        if (
            self._initial_nnf_tensor is None
            or self._initial_nnf_shape != initial_nnf_np.shape
        ):
            self._initial_nnf_shape = initial_nnf_np.shape
            self._initial_nnf_tensor = torch.empty(
                initial_nnf_np.shape,
                dtype=torch.int32,
                device=self.engine.device,
            )

        self._initial_nnf_tensor.copy_(
            torch.from_numpy(initial_nnf_np),
            non_blocking=True,
        )
        return self._initial_nnf_tensor

    def _build_pyramid_shapes(self) -> List[Tuple[int, int, int, int]]:
        shapes = []
        for i in range(self.num_pyramid_levels):
            scale = 2.0 ** -(self.num_pyramid_levels - 1 - i)
            shapes.append(
                (
                    max(1, int(self.sh * scale)),
                    max(1, int(self.sw * scale)),
                    max(1, int(self.th * scale)),
                    max(1, int(self.tw * scale)),
                )
            )
        return shapes

    def _validate_targets(self, guides: List[GuideObject]) -> None:
        _validate_guides(guides)
        if len(guides) != len(self.guide_channel_counts):
            raise ValueError(
                f"Expected {len(self.guide_channel_counts)} guides, got {len(guides)}."
            )
        for i, (guide, channels) in enumerate(zip(guides, self.guide_channel_counts)):
            if guide.target.shape[:2] != (self.th, self.tw):
                raise ValueError(
                    f"guides[{i}] target shape changed from {(self.th, self.tw)} "
                    f"to {guide.target.shape[:2]}."
                )
            if guide.target.shape[2] != channels:
                raise ValueError(
                    f"guides[{i}] target channel count changed from {channels} "
                    f"to {guide.target.shape[2]}."
                )


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
            if CudaBackend is None:
                raise RuntimeError(
                    "CUDA backend requested but ebsynth_torch extension is not available. "
                    "The extension will be JIT compiled on first run. "
                    "If you see this error, the JIT compilation may have failed. "
                    "Check the error output above for compilation details."
                )
            # Check if device is specified in config (for CPU mode with C++ extension)
            device = getattr(ebsynth_config, "device", None)
            self.backend = CudaBackend(ebsynth_config, pipeline_config, device=device)
        elif self.backend_type == "torch":
            self.backend = PyTorchBackend(ebsynth_config, pipeline_config)
        elif self.backend_type == "taichi":
            if TaichiBackend is None:
                raise RuntimeError(
                    "Taichi backend requested but Taichi is not available. "
                    "Please install taichi: pip install taichi"
                )
            self.backend = TaichiBackend(ebsynth_config, pipeline_config)
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
        guides: List[GuideObject],
        modulation_map: Optional[np.ndarray] = None,
        initial_nnf: Optional[np.ndarray] = None,
        return_error: bool = True,
        output_nnf: bool = False,
        benchmark: bool = False,
    ) -> EngineRunResult:
        """Runs the full pyramidal synthesis process."""
        if benchmark:
            self._enable_benchmarking()

        context = PreparedSynthesisContext(self, style_img, guides)
        result = context.run_frame(
            guides=guides,
            modulation_map=modulation_map,
            initial_nnf=initial_nnf,
            return_error=return_error,
            output_nnf=output_nnf,
        )

        if benchmark:
            self._print_benchmark_summary()

        return result

    def _enable_benchmarking(self) -> None:
        self.benchmark_enabled = True
        self.timer.reset()
        if hasattr(self.backend, "enable_benchmarking"):
            self.backend.enable_benchmarking(True)

    def _ensure_rand_states(self, th: int, tw: int) -> None:
        if (
            self.rand_states is None
            or self.rand_states.numel() * self.rand_states.element_size() < th * tw * 48
        ):
            self.rand_states = torch.empty(
                th * tw * 48, dtype=torch.uint8, device=self.device
            )
            if self.backend_type == "cuda":
                ebsynth_torch.init_rand_states(self.rand_states)

    def _determine_num_pyramid_levels(
        self, sh: int, sw: int, th: int, tw: int
    ) -> int:
        max_levels = 0
        min_dim_start = min(sh, sw, th, tw)
        for level in range(32, -1, -1):
            if (min_dim_start * (2.0**-level)) >= (
                2 * self.ebsynth_config.patch_size + 1
            ):
                max_levels = level + 1
                break
        return min(self.pipeline_config.pyramid_levels, max_levels)

    def _run_prepared_context(
        self,
        *,
        context: PreparedSynthesisContext,
        target_guide_cat: torch.Tensor,
        modulation_tensor: torch.Tensor,
        p_target_guide: List[torch.Tensor],
        p_modulation: List[torch.Tensor],
        initial_nnf: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nnf = None
        output_image = None
        output_error = None
        vote_mode = self.vote_mode_map[self.ebsynth_config.vote_mode]
        cost_function_mode = self.cost_function_map[self.ebsynth_config.cost_function]

        def _run_pyramid_loop():
            nonlocal nnf, output_image, output_error
            nnf = None
            output_image = None
            output_error = None

            for level in range(context.num_pyramid_levels):

                def _process_pyramid_level():
                    nonlocal nnf, output_image, output_error
                    p_style_level = context.p_style[level]
                    p_source_guide_level = context.p_source_guide[level]
                    p_target_guide_level = p_target_guide[level]
                    p_modulation_level = p_modulation[level]

                    p_sh, p_sw, _ = p_style_level.shape
                    p_th, p_tw, _ = p_target_guide_level.shape

                    if level == 0:
                        if initial_nnf is not None:
                            initial_nnf_tensor = context.initial_nnf_tensor(initial_nnf)
                            scale_h = p_th / context.th
                            scale_w = p_tw / context.tw

                            nnf_float = (
                                initial_nnf_tensor.permute(2, 0, 1).unsqueeze(0).float()
                            )
                            resampled_nnf = F.interpolate(
                                nnf_float,
                                size=(p_th, p_tw),
                                mode="bilinear",
                                align_corners=False,
                            )
                            resampled_nnf[:, 0, :, :] *= scale_w
                            resampled_nnf[:, 1, :, :] *= scale_h
                            nnf = (
                                resampled_nnf.squeeze(0)
                                .permute(1, 2, 0)
                                .to(torch.int32)
                                .contiguous()
                            )
                        else:
                            nnf = random_init_nnf(
                                self.device,
                                p_th,
                                p_tw,
                                p_sh,
                                p_sw,
                                self.ebsynth_config.patch_size,
                            )
                    else:
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

                    output_image, output_error, nnf = self.backend.run_level(
                        p_style_level,
                        p_source_guide_level,
                        p_target_guide_level,
                        p_modulation_level,
                        nnf,
                        context.style_weights,
                        context.guide_weights,
                        self.ebsynth_config.uniformity,
                        self.ebsynth_config.patch_size,
                        vote_mode,
                        self.ebsynth_config.search_vote_iters,
                        self.ebsynth_config.patch_match_iters,
                        self.ebsynth_config.stop_threshold,
                        self.rand_states,
                        cost_function_mode,
                        self.benchmark_enabled,
                    )

                self._timed_operation(f"pyramid_level_{level}", _process_pyramid_level)

        self._timed_operation("pyramid_processing", _run_pyramid_loop)

        if self.ebsynth_config.extra_pass_3x3:
            print("  - Performing final 3x3 pass...")
            output_image, output_error, nnf = self.backend.run_level(
                context.style_tensor,
                context.source_guide_cat,
                target_guide_cat,
                modulation_tensor,
                nnf,
                context.style_weights,
                context.guide_weights,
                0.0,
                3,
                vote_mode,
                self.ebsynth_config.search_vote_iters,
                self.ebsynth_config.patch_match_iters,
                self.ebsynth_config.stop_threshold,
                self.rand_states,
                cost_function_mode,
                self.benchmark_enabled,
            )

        return output_image, output_error, nnf

    def _print_benchmark_summary(self) -> None:
        print("\n" + "=" * 60)
        print("SYNTHESIS ENGINE TIMING SUMMARY")
        print("=" * 60)
        self.timer.print_summary("Synthesis Engine Operations")

        if hasattr(self.backend, "timer"):
            print("\n" + "=" * 60)
            backend_name = (
                "CUDA"
                if self.backend_type == "cuda"
                else ("Taichi" if self.backend_type == "taichi" else "PyTorch")
            )
            print(f"{backend_name.upper()} BACKEND TIMING SUMMARY")
            print("=" * 60)
            self.backend.timer.print_summary(f"{backend_name} Backend Operations")


def _channel_weights(num_channels: int, total_weight: float, device: str) -> torch.Tensor:
    if num_channels <= 0:
        raise ValueError("Expected at least one channel when building weights.")
    return torch.full(
        (num_channels,),
        float(total_weight) / float(num_channels),
        dtype=torch.float32,
        device=device,
    )


def _guide_weights(
    guide_sources: List[np.ndarray], guide_weights: List[float], device: str
) -> torch.Tensor:
    weights = []
    for guide, weight in zip(guide_sources, guide_weights):
        if guide.ndim != 3:
            raise ValueError(f"Guide arrays must be HWC, got shape {guide.shape}.")
        weights.extend([float(weight) / guide.shape[2]] * guide.shape[2])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _channel_slices(channel_counts: List[int]) -> List[slice]:
    slices = []
    start = 0
    for channels in channel_counts:
        stop = start + channels
        slices.append(slice(start, stop))
        start = stop
    return slices


def _contiguous_array(array: np.ndarray) -> np.ndarray:
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _validate_image_array(array: np.ndarray, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if array.ndim != 3:
        raise ValueError(f"{name} must be an HWC image, got shape {array.shape}.")
    if array.shape[2] <= 0:
        raise ValueError(f"{name} must have at least one channel.")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}.")


def _validate_guides(guides: List[GuideObject]) -> None:
    if not guides:
        raise ValueError("At least one guide is required.")
    for i, guide in enumerate(guides):
        if not isinstance(guide, GuideObject):
            raise TypeError(f"guides[{i}] must be a GuideObject.")
        _validate_image_array(guide.keyframe, f"guides[{i}].keyframe")
        _validate_image_array(guide.target, f"guides[{i}].target")
        if guide.keyframe.shape[2] != guide.target.shape[2]:
            raise ValueError(
                f"guides[{i}] keyframe/target channel mismatch: "
                f"{guide.keyframe.shape[2]} vs {guide.target.shape[2]}"
            )
