"""ComfyUI nodes for video EbSynth synthesis."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Any

import torch

from .image_io import (
    bgr_uint8_sequence_to_image_tensor,
    image_tensor_to_rgb_float_sequence,
)
from .types import ReEzVideoStyleKeyframes, ReEzVideoSynthConfig

REEZ_VIDEO_STYLES = "REEZ_VIDEO_STYLES"
REEZ_VIDEO_SYNTH_CONFIG = "REEZ_VIDEO_SYNTH_CONFIG"

_FLOW_ENGINES = ("NeuFlow", "RAFT", "OpenCV", "TorchVision")
_FLOW_MODELS = ("neuflow_mixed", "neuflow_sintel", "neuflow_things", "sintel", "kitti", "small")
_EDGE_METHODS = ("Classic", "PAGE", "PST")
_OPENCV_FLOW_METHODS = ("DIS", "FARNEBACK")
_TORCHVISION_FLOW_MODELS = ("raft_large", "raft_small")
_POISSON_SOLVERS = ("disabled", "lsqr", "lsmr", "cg", "amg", "seamless", "taichi-cg")
_BACKENDS = ("cuda", "torch", "taichi")
_COST_FUNCTIONS = ("ssd", "ncc")
_VOTE_MODES = ("weighted", "plain")


def _parse_indices(style_indices: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in style_indices.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("style_indices must be comma-separated frame numbers, e.g. 0, 24, 48.") from exc
    if not indices:
        raise ValueError("At least one style index is required.")
    if any(index < 0 for index in indices):
        raise ValueError("style_indices cannot contain negative frame numbers.")
    return indices


def _make_style_keyframes(
    frames: list[Any],
    indices: list[int],
) -> ReEzVideoStyleKeyframes:
    if len(frames) != len(indices):
        raise ValueError(
            f"Expected the same number of style images and indices, got {len(frames)} images and {len(indices)} indices."
        )
    if len(set(indices)) != len(indices):
        raise ValueError("Each style keyframe index must be unique.")

    ordered = sorted(zip(indices, frames), key=lambda item: item[0])
    return ReEzVideoStyleKeyframes(
        frames=[frame for _, frame in ordered],
        indices=[index for index, _ in ordered],
    )


def _path_or_none(path: str | None) -> str | None:
    if path is None:
        return None
    stripped = path.strip()
    return stripped or None


def _build_configs(cfg: ReEzVideoSynthConfig):
    from ezsynth.config import (
        BlendingConfig,
        DebugConfig,
        EbsynthParamsConfig,
        PipelineConfig,
        PrecomputationConfig,
    )

    poisson_maxiter = cfg.poisson_maxiter if cfg.poisson_maxiter > 0 else None
    precomputation = PrecomputationConfig(
        flow_engine=cfg.flow_engine,
        flow_model=cfg.flow_model,
        edge_method=cfg.edge_method,
        opencv_flow_method=cfg.opencv_flow_method,
        torchvision_flow_model=cfg.torchvision_flow_model,
    )
    pipeline = PipelineConfig(
        pyramid_levels=cfg.pyramid_levels,
        use_residual_transfer=cfg.use_residual_transfer,
        alpha=cfg.alpha,
        use_temporal_nnf_propagation=cfg.use_temporal_nnf_propagation,
        use_sparse_feature_guide=cfg.use_sparse_feature_guide,
        use_pseudo_endpoint_styles=cfg.use_pseudo_endpoint_styles,
        use_flow_occlusion_masks=cfg.use_flow_occlusion_masks,
        use_forward_warping=cfg.use_forward_warping,
    )
    blending = BlendingConfig(
        poisson_solver=cfg.poisson_solver,
        poisson_maxiter=poisson_maxiter,
        poisson_grad_weight_l=cfg.poisson_grad_weight_l,
        poisson_grad_weight_ab=cfg.poisson_grad_weight_ab,
        use_taichi_ops=cfg.use_taichi_ops,
        use_forward_warping=cfg.use_forward_warping,
    )
    ebsynth_params = EbsynthParamsConfig(
        uniformity=cfg.uniformity,
        patch_size=cfg.patch_size,
        vote_mode=cfg.vote_mode,
        search_vote_iters=cfg.search_vote_iters,
        patch_match_iters=cfg.patch_match_iters,
        stop_threshold=cfg.stop_threshold,
        search_pruning_threshold=cfg.search_pruning_threshold,
        use_bilateral=cfg.use_bilateral,
        sigma_spatial=cfg.sigma_spatial,
        sigma_color=cfg.sigma_color,
        n_size_step=cfg.n_size_step,
        cost_function=cfg.cost_function,
        backend=cfg.backend,
        extra_pass_3x3=cfg.extra_pass_3x3,
        edge_weight=cfg.edge_weight,
        image_weight=cfg.image_weight,
        pos_weight=cfg.pos_weight,
        warp_weight=cfg.warp_weight,
        sparse_anchor_weight=cfg.sparse_anchor_weight,
        use_optimization=cfg.use_optimization,
    )
    debug = DebugConfig(save_flow_viz=cfg.save_flow_viz)
    return precomputation, pipeline, blending, ebsynth_params, debug


def _fps_from_video_info(video_info: dict[str, Any] | None, fallback: float) -> float:
    if video_info:
        fps = video_info.get("loaded_fps") or video_info.get("source_fps")
        if fps:
            return float(fps)
    return float(fallback)


def _build_video_info(
    video_info: dict[str, Any] | None,
    *,
    frame_count: int,
    frame_rate: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    duration = frame_count / frame_rate if frame_rate > 0 else 0.0
    info = dict(video_info or {})
    for key, value in {
        "source_fps": frame_rate,
        "source_frame_count": frame_count,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
    }.items():
        info.setdefault(key, value)
    info.update(
        {
            "loaded_fps": frame_rate,
            "loaded_frame_count": frame_count,
            "loaded_duration": duration,
            "loaded_width": width,
            "loaded_height": height,
        }
    )
    return info


class ReEzSynthVideoStyleKeyframe:
    """Create one style keyframe for the video EbSynth pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "style_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("style",)
    FUNCTION = "create"
    CATEGORY = "ReEzSynth/video"

    def create(self, style_image, style_index: int):
        frames = image_tensor_to_rgb_float_sequence(style_image)
        if len(frames) != 1:
            raise ValueError(
                f"Video Style Keyframe expects a single image, got a batch of {len(frames)}."
            )
        return (_make_style_keyframes(frames, [int(style_index)]),)


class ReEzSynthVideoStyleKeyframes:
    """Pair a batch of style keyframes with their content frame indices."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_images": ("IMAGE",),
                "style_indices": ("STRING", {"default": "0"}),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("styles",)
    FUNCTION = "create"
    CATEGORY = "ReEzSynth/video"

    def create(self, style_images, style_indices: str):
        frames = image_tensor_to_rgb_float_sequence(style_images)
        indices = _parse_indices(style_indices)
        return (_make_style_keyframes(frames, indices),)


class ReEzSynthVideoStyleKeyframesAppend:
    """Append style keyframes to an existing video style list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (REEZ_VIDEO_STYLES,),
            },
            "optional": {
                "styles": (REEZ_VIDEO_STYLES,),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("styles",)
    FUNCTION = "append"
    CATEGORY = "ReEzSynth/video"

    def append(
        self,
        style: ReEzVideoStyleKeyframes,
        styles: ReEzVideoStyleKeyframes | None = None,
    ):
        base_frames = styles.frames if styles is not None else []
        base_indices = styles.indices if styles is not None else []
        return (
            _make_style_keyframes(
                frames=[*base_frames, *style.frames],
                indices=[*base_indices, *style.indices],
            ),
        )


class ReEzSynthVideoSynthConfig:
    """Parameters for the full video EbSynth pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_engine": (list(_FLOW_ENGINES), {"default": "RAFT"}),
                "flow_model": (list(_FLOW_MODELS), {"default": "sintel"}),
                "edge_method": (list(_EDGE_METHODS), {"default": "Classic"}),
                "opencv_flow_method": (list(_OPENCV_FLOW_METHODS), {"default": "DIS"}),
                "torchvision_flow_model": (list(_TORCHVISION_FLOW_MODELS), {"default": "raft_large"}),
                "pyramid_levels": ("INT", {"default": 6, "min": 1, "max": 12, "step": 1}),
                "alpha": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_residual_transfer": ("BOOLEAN", {"default": False}),
                "use_temporal_nnf_propagation": ("BOOLEAN", {"default": False}),
                "use_sparse_feature_guide": ("BOOLEAN", {"default": False}),
                "use_pseudo_endpoint_styles": ("BOOLEAN", {"default": False}),
                "use_flow_occlusion_masks": ("BOOLEAN", {"default": False}),
                "use_forward_warping": ("BOOLEAN", {"default": False}),
                "poisson_solver": (list(_POISSON_SOLVERS), {"default": "disabled"}),
                "poisson_maxiter": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "poisson_grad_weight_l": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "poisson_grad_weight_ab": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "use_taichi_ops": ("BOOLEAN", {"default": False}),
                "backend": (list(_BACKENDS), {"default": "taichi"}),
                "cost_function": (list(_COST_FUNCTIONS), {"default": "ncc"}),
                "vote_mode": (list(_VOTE_MODES), {"default": "weighted"}),
                "uniformity": ("FLOAT", {"default": 3500.0, "min": 0.0, "max": 20000.0, "step": 10.0}),
                "patch_size": ("INT", {"default": 7, "min": 3, "max": 31, "step": 2}),
                "search_vote_iters": ("INT", {"default": 12, "min": 1, "max": 64, "step": 1}),
                "patch_match_iters": ("INT", {"default": 6, "min": 1, "max": 32, "step": 1}),
                "stop_threshold": ("INT", {"default": 5, "min": 0, "max": 255, "step": 1}),
                "search_pruning_threshold": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 10000.0, "step": 1.0},
                ),
                "use_bilateral": ("BOOLEAN", {"default": False}),
                "sigma_spatial": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 64.0, "step": 0.1}),
                "sigma_color": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 255.0, "step": 0.1}),
                "n_size_step": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "extra_pass_3x3": ("BOOLEAN", {"default": False}),
                "image_weight": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "edge_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "warp_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sparse_anchor_weight": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 200.0, "step": 0.1},
                ),
                "use_optimization": ("BOOLEAN", {"default": True}),
                "save_flow_viz": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_SYNTH_CONFIG,)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "ReEzSynth/video"

    def build(self, **kwargs):
        if kwargs["patch_size"] % 2 == 0:
            raise ValueError("patch_size must be odd.")
        return (ReEzVideoSynthConfig(**kwargs),)


class ReEzSynthVideoSynth:
    """Run EbSynth over a video/image batch and return VHS-compatible outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_images": ("IMAGE",),
                "styles": (REEZ_VIDEO_STYLES,),
                "config": (REEZ_VIDEO_SYNTH_CONFIG,),
                "frame_rate": ("FLOAT", {"default": 8.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "save_outputs": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
                "output_dir": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("images", "frame_count", "frame_rate", "audio", "video_info")
    FUNCTION = "synthesize"
    CATEGORY = "ReEzSynth/video"

    def synthesize(
        self,
        content_images,
        styles: ReEzVideoStyleKeyframes,
        config: ReEzVideoSynthConfig,
        frame_rate: float,
        save_outputs: bool,
        audio=None,
        video_info: dict[str, Any] | None = None,
        output_dir: str = "",
        cache_dir: str = "",
    ):
        from ezsynth.integration_io import write_rgb_frame_sequence, write_rgb_style_images
        from ezsynth.service import SynthesisRequest, SynthesisService

        content_frames = image_tensor_to_rgb_float_sequence(content_images)
        if not content_frames:
            raise ValueError("content_images must contain at least one frame.")
        if any(index >= len(content_frames) for index in styles.indices):
            raise ValueError(
                f"style_indices must be within content frame range 0-{len(content_frames) - 1}."
            )

        resolved_fps = _fps_from_video_info(video_info, frame_rate)
        precomputation, pipeline, blending, ebsynth_params, debug = _build_configs(config)
        out_dir = _path_or_none(output_dir)
        cache = _path_or_none(cache_dir)

        with tempfile.TemporaryDirectory(prefix="reezsynth_comfy_video_") as temp_root:
            root = Path(temp_root)
            content_paths = write_rgb_frame_sequence(content_frames, root / "content")
            style_paths = write_rgb_style_images(styles.frames, root / "styles")
            request = SynthesisRequest(
                content_dir=str(content_paths.directory),
                style_paths=style_paths,
                style_indices=styles.indices,
                output_dir=out_dir,
                cache_dir=cache,
                project_name="ReEzSynthComfyVideo",
                save_outputs=save_outputs,
                ebsynth_params=ebsynth_params,
                pipeline=pipeline,
                blending=blending,
                precomputation=precomputation,
                debug=debug,
            )
            result = SynthesisService().run(request)

        images = bgr_uint8_sequence_to_image_tensor(result.frames)
        frame_count = int(images.shape[0])
        height = int(images.shape[1])
        width = int(images.shape[2])
        out_video_info = _build_video_info(
            video_info,
            frame_count=frame_count,
            frame_rate=resolved_fps,
            width=width,
            height=height,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (images, frame_count, resolved_fps, audio, out_video_info)
