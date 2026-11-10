"""ComfyUI nodes for video EbSynth synthesis (``ezsynth.pipeline.SynthesisPipeline``)."""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import bgr_uint8_sequence_to_image_tensor, image_batch_to_bgr_uint8_list
from .nodes_image_synth import (
    BACKEND_TOOLTIP,
    COST_TOOLTIP,
    MULTICAND_TOOLTIP,
    PRECISION_TOOLTIP,
    SEARCH_MODE_TOOLTIP,
    _bool,
    _combo,
    _float,
    _int,
    synthesis_core_inputs,
)
from .options import (
    COST_FUNCTIONS,
    EDGE_METHODS,
    FLOW_ENGINES,
    FLOW_MODELS,
    MULTICAND_K_RANGE,
    OPENCV_FLOW_METHODS,
    POISSON_SOLVERS,
    PRECISIONS,
    SEARCH_MODES,
    SYNTH_BACKENDS,
    TORCHVISION_FLOW_MODELS,
    VOTE_MODES,
    check_indices_in_range,
    check_patch_size,
    check_synthesis_options,
    parse_indices,
    require_module,
    validate_synthesis_options,
)
from .types import (
    REEZ_VIDEO_STYLES,
    REEZ_VIDEO_SYNTH_CONFIG,
    ReEzVideoStyleKeyframes,
    ReEzVideoSynthConfig,
)

CATEGORY = "ReEzSynth/video"


def make_style_keyframes(
    frames: list[np.ndarray], indices: list[int]
) -> ReEzVideoStyleKeyframes:
    """Pair style frames with content indices, sorted by index."""
    if len(frames) != len(indices):
        raise ValueError(
            f"Got {len(frames)} style images but {len(indices)} indices; "
            "give one index per style image."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Each style keyframe index must be unique (got {indices}).")
    order = sorted(range(len(indices)), key=indices.__getitem__)
    return ReEzVideoStyleKeyframes(
        frames=tuple(frames[i] for i in order),
        indices=tuple(indices[i] for i in order),
    )


def build_video_configs(cfg: ReEzVideoSynthConfig, device: str | None):
    """Translate the node config into the ezsynth pydantic config sections."""
    from ezsynth.config import (
        BlendingConfig,
        DebugConfig,
        EbsynthParamsConfig,
        PipelineConfig,
        PrecomputationConfig,
    )

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
        poisson_maxiter=cfg.poisson_maxiter if cfg.poisson_maxiter > 0 else None,
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
        search_mode=cfg.search_mode,
        multicand_k=cfg.multicand_k,
        precision=cfg.precision,
        backend=cfg.backend,
        device=device,
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


def check_video_dependencies(cfg: ReEzVideoSynthConfig) -> None:
    """Fail before any work when the chosen options need a missing optional package."""
    check_synthesis_options(
        cfg.backend, cfg.cost_function, cfg.search_mode, cfg.precision, cfg.multicand_k
    )
    if cfg.edge_method in ("PAGE", "PST"):
        require_module("phycv")
    if cfg.poisson_solver == "amg":
        require_module("pyamg")
    if cfg.poisson_solver == "taichi-cg" or cfg.use_taichi_ops:
        require_module("taichi")
    if cfg.flow_engine == "NeuFlow":
        require_module("huggingface_hub")


def build_video_info(
    video_info: dict[str, Any] | None,
    *,
    frame_count: int,
    frame_rate: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    """VHS-compatible video info: incoming ``source_*`` kept, ``loaded_*`` = output."""
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


def _in_memory_project_data(project_cfg, content_frames, style_frames):
    """ProjectData serving frames already in memory (no PNG round trip)."""
    from ezsynth.data import ProjectData
    from ezsynth.utils.image_utils import (
        assert_uniform_image_shapes,
        resize_image_to_match,
    )

    assert_uniform_image_shapes(content_frames)
    styles = [resize_image_to_match(style, content_frames[0]) for style in style_frames]

    class _InMemoryProjectData(ProjectData):
        def get_content_frames(self, force_reload: bool = False) -> list[np.ndarray]:
            return content_frames

        def get_style_frames(self, force_reload: bool = False) -> list[np.ndarray]:
            return styles

    return _InMemoryProjectData(project_cfg)


class ReEzSynthVideoStyleKeyframe:
    DESCRIPTION = "One stylized keyframe and the content frame it corresponds to."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": (
                    "IMAGE",
                    {"tooltip": "Stylized version of content frame style_index."},
                ),
                "style_index": _int(
                    0,
                    0,
                    1000000,
                    1,
                    "0-based index of the content frame this style matches.",
                ),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("style",)
    OUTPUT_TOOLTIPS = ("Style keyframe list with one entry.",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, style_image, style_index: int):
        frames = image_batch_to_bgr_uint8_list(style_image)
        if len(frames) != 1:
            raise ValueError(
                f"Video Style Keyframe expects a single image, got a batch of {len(frames)}. "
                "Use Video Style Keyframes for batches."
            )
        return (make_style_keyframes(frames, [int(style_index)]),)


class ReEzSynthVideoStyleKeyframes:
    DESCRIPTION = "A batch of stylized keyframes with their content frame indices."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_images": (
                    "IMAGE",
                    {
                        "tooltip": "Batch of stylized keyframes, in the order of style_indices."
                    },
                ),
                "style_indices": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "Content frame index of each style image, e.g. '0, 24, 48'.",
                    },
                ),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("styles",)
    OUTPUT_TOOLTIPS = ("Style keyframe list.",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, style_images, style_indices: str):
        frames = image_batch_to_bgr_uint8_list(style_images)
        return (
            make_style_keyframes(frames, parse_indices(style_indices, "style_indices")),
        )


class ReEzSynthVideoStyleKeyframesAppend:
    DESCRIPTION = "Merge style keyframe lists."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (REEZ_VIDEO_STYLES, {"tooltip": "Keyframe(s) to add."}),
            },
            "optional": {
                "styles": (
                    REEZ_VIDEO_STYLES,
                    {
                        "tooltip": "Existing keyframe list to extend (leave empty to start one)."
                    },
                ),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_STYLES,)
    RETURN_NAMES = ("styles",)
    OUTPUT_TOOLTIPS = ("Merged keyframe list, sorted by frame index.",)
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(
        self,
        style: ReEzVideoStyleKeyframes,
        styles: ReEzVideoStyleKeyframes | None = None,
    ):
        base_frames = styles.frames if styles is not None else ()
        base_indices = styles.indices if styles is not None else ()
        return (
            make_style_keyframes(
                [*base_frames, *style.frames], [*base_indices, *style.indices]
            ),
        )


class ReEzSynthVideoSynthConfig:
    DESCRIPTION = (
        "Parameters for ReEzSynth Video Synth (flow, edges, blending, EbSynth)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        core = synthesis_core_inputs()
        return {
            # Widget order is part of saved workflows: append new inputs at the end.
            "required": {
                "flow_engine": _combo(
                    FLOW_ENGINES,
                    "RAFT",
                    "Optical flow: RAFT (raft-*.pth in models/raft, see "
                    "models/model_files.md), NeuFlow (needs huggingface-hub; "
                    "neuflow_*.pth in models/neuflow), OpenCV (CPU, no weights), "
                    "TorchVision RAFT (weights download automatically).",
                ),
                "flow_model": _combo(
                    FLOW_MODELS,
                    "sintel",
                    "Checkpoint for the flow engine: sintel/kitti/small for RAFT, "
                    "neuflow_* for NeuFlow.",
                ),
                "edge_method": _combo(
                    EDGE_METHODS,
                    "Classic",
                    "Edge guide: Classic (OpenCV), PAGE / PST (need `pip install phycv`).",
                ),
                "opencv_flow_method": _combo(
                    OPENCV_FLOW_METHODS,
                    "DIS",
                    "OpenCV flow algorithm (flow_engine OpenCV).",
                ),
                "torchvision_flow_model": _combo(
                    TORCHVISION_FLOW_MODELS,
                    "raft_large",
                    "TorchVision RAFT variant (flow_engine TorchVision).",
                ),
                "pyramid_levels": _int(6, 1, 12, 1, "Coarse-to-fine pyramid levels."),
                "alpha": _float(
                    0.75,
                    0.0,
                    1.0,
                    0.01,
                    "Blend weight between forward and backward passes near keyframes.",
                ),
                "use_residual_transfer": _bool(
                    False,
                    "Transfer the style's high-frequency residual for crisper detail.",
                ),
                "use_temporal_nnf_propagation": _bool(
                    False,
                    "Seed each frame's search with the previous frame's NNF (more stable, faster).",
                ),
                "use_sparse_feature_guide": _bool(
                    False, "Add a sparse feature-track guide to pin moving structures."
                ),
                "use_pseudo_endpoint_styles": _bool(
                    False,
                    "Synthesize extra style endpoints when only one keyframe bounds a segment.",
                ),
                "use_flow_occlusion_masks": _bool(
                    False,
                    "Mask occluded regions from forward/backward flow consistency.",
                ),
                "use_forward_warping": _bool(
                    False, "Forward-splat the warp guide instead of backward warping."
                ),
                "poisson_solver": _combo(
                    POISSON_SOLVERS,
                    "disabled",
                    "Gradient-domain blending of forward/backward passes. amg needs pyamg, "
                    "taichi-cg needs taichi; disabled uses histogram blending only.",
                ),
                "poisson_maxiter": _int(
                    0, 0, 10000, 1, "Max solver iterations (0 = solver default)."
                ),
                "poisson_grad_weight_l": _float(
                    2.5, 0.0, 20.0, 0.1, "Poisson gradient weight for luminance."
                ),
                "poisson_grad_weight_ab": _float(
                    0.5, 0.0, 20.0, 0.1, "Poisson gradient weight for chroma."
                ),
                "use_taichi_ops": _bool(
                    False, "Run blending color ops with Taichi (needs taichi)."
                ),
                "backend": _combo(SYNTH_BACKENDS, "taichi", BACKEND_TOOLTIP),
                "cost_function": _combo(COST_FUNCTIONS, "ncc", COST_TOOLTIP),
                "vote_mode": _combo(
                    VOTE_MODES,
                    "weighted",
                    "Patch voting: weighted by match error, or plain average.",
                ),
                "uniformity": core["uniformity"],
                "patch_size": core["patch_size"],
                "search_vote_iters": core["search_vote_iters"],
                "patch_match_iters": core["patch_match_iters"],
                "stop_threshold": _int(
                    5,
                    0,
                    255,
                    1,
                    "Skip refining patches whose error is already below this.",
                ),
                "search_pruning_threshold": _float(
                    50.0,
                    0.0,
                    10000.0,
                    1.0,
                    "Prune random-search candidates worse than this.",
                ),
                "use_bilateral": _bool(
                    False, "Edge-aware (bilateral) weighting of patch votes."
                ),
                "sigma_spatial": _float(
                    4.0, 0.1, 64.0, 0.1, "Bilateral spatial sigma."
                ),
                "sigma_color": _float(10.0, 0.1, 255.0, 0.1, "Bilateral color sigma."),
                "n_size_step": _int(
                    1, 1, 8, 1, "Neighbour step used by the bilateral filter."
                ),
                "extra_pass_3x3": _bool(
                    False, "Extra 3x3 refinement pass at the finest level."
                ),
                "image_weight": _float(
                    6.0, 0.0, 100.0, 0.1, "Weight of the content image guide."
                ),
                "edge_weight": _float(
                    1.0, 0.0, 100.0, 0.1, "Weight of the edge guide."
                ),
                "pos_weight": _float(
                    2.0, 0.0, 100.0, 0.1, "Weight of the positional guide."
                ),
                "warp_weight": _float(
                    0.5, 0.0, 100.0, 0.1, "Weight of the flow-warped style guide."
                ),
                "sparse_anchor_weight": _float(
                    50.0, 0.0, 200.0, 0.1, "Weight of the sparse feature guide."
                ),
                "use_optimization": _bool(
                    True, "Use the optimized search/vote kernels (same result, faster)."
                ),
                "save_flow_viz": _bool(
                    False,
                    "Write flow visualizations to debug/flow_viz (debugging only).",
                ),
                "search_mode": _combo(SEARCH_MODES, "patchmatch", SEARCH_MODE_TOOLTIP),
                "multicand_k": _int(3, *MULTICAND_K_RANGE, 1, MULTICAND_TOOLTIP),
                "precision": _combo(PRECISIONS, "float32", PRECISION_TOOLTIP),
            },
        }

    RETURN_TYPES = (REEZ_VIDEO_SYNTH_CONFIG,)
    RETURN_NAMES = ("config",)
    OUTPUT_TOOLTIPS = ("Config for ReEzSynth Video Synth.",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        backend=None,
        cost_function=None,
        search_mode=None,
        precision=None,
        multicand_k=None,
    ):
        values = (backend, cost_function, search_mode, precision, multicand_k)
        if any(v is None for v in values):
            return True  # linked inputs: validated in build()
        return validate_synthesis_options(*values) or True

    def build(self, **kwargs):
        cfg = ReEzVideoSynthConfig(**kwargs)
        check_patch_size(cfg.patch_size)
        check_video_dependencies(cfg)
        return (cfg,)


class ReEzSynthVideoSynth:
    DESCRIPTION = (
        "Propagate stylized keyframes through a frame batch with the full EbSynth "
        "video pipeline (optical flow, edge guides, forward/backward passes, blending)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_images": (
                    "IMAGE",
                    {"tooltip": "All content frames as one IMAGE batch."},
                ),
                "styles": (
                    REEZ_VIDEO_STYLES,
                    {"tooltip": "Stylized keyframes with their frame indices."},
                ),
                "config": (
                    REEZ_VIDEO_SYNTH_CONFIG,
                    {"tooltip": "From ReEzSynth Video Synth Config."},
                ),
                "frame_rate": _float(
                    8.0,
                    0.001,
                    240.0,
                    0.001,
                    "Frame rate reported on the outputs (connect the source video fps).",
                ),
                "save_outputs": _bool(
                    False, "Also write the frames as PNGs into output_dir."
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Passed through unchanged to the audio output."},
                ),
                "video_info": (
                    "VHS_VIDEOINFO",
                    {
                        "tooltip": "VideoHelperSuite video info; passed through with loaded_* updated."
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder for save_outputs (required when saving).",
                    },
                ),
                "cache_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder to cache optical flow / edges between runs. Empty: "
                        "temporary. Use one folder per clip (the cache is not keyed by content).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("images", "frame_count", "frame_rate", "audio", "video_info")
    OUTPUT_TOOLTIPS = (
        "Stylized frames.",
        "Number of output frames.",
        "The frame_rate input.",
        "The audio input, unchanged.",
        "Video info describing the output frames.",
    )
    FUNCTION = "synthesize"
    CATEGORY = CATEGORY

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
        check_video_dependencies(config)
        out_dir = output_dir.strip()
        if save_outputs and not out_dir:
            raise ValueError("save_outputs is on but output_dir is empty.")

        from ezsynth.config import ProjectConfig
        from ezsynth.output import OutputManager
        from ezsynth.service import SynthesisConfigs, SynthesisService

        from . import comfy_runtime as rt

        content_frames = image_batch_to_bgr_uint8_list(content_images)
        check_indices_in_range(
            list(styles.indices), len(content_frames), "style_indices"
        )

        device = rt.torch_device()
        height, width = content_frames[0].shape[:2]
        precomputation, pipeline, blending, ebsynth_params, debug = build_video_configs(
            config, str(device)
        )
        rt.prepare_vram(device, rt.estimate_synthesis_bytes(height, width, 12))

        with ExitStack() as stack:
            cache = cache_dir.strip() or stack.enter_context(
                tempfile.TemporaryDirectory(prefix="reezsynth_cache_")
            )
            work_out = out_dir or stack.enter_context(
                tempfile.TemporaryDirectory(prefix="reezsynth_out_")
            )
            project = ProjectConfig(
                name="ReEzSynthComfyVideo",
                content_dir="<comfyui images>",
                style_path=[f"<comfyui style {i}>" for i in styles.indices],
                style_indices=list(styles.indices),
                output_dir=work_out,
                cache_dir=cache,
            )
            configs = SynthesisConfigs(
                project=project,
                precomputation=precomputation,
                pipeline=pipeline,
                blending=blending,
                ebsynth_params=ebsynth_params,
                debug=debug,
            )
            data = _in_memory_project_data(project, content_frames, list(styles.frames))
            try:
                with rt.comfy_progress():
                    frames = SynthesisService().build_pipeline(configs, data).run()
            finally:
                rt.release_vram()
            if save_outputs:
                OutputManager(data).handle(
                    frames, save=True, visible_output_dir=Path(out_dir)
                )

        images = bgr_uint8_sequence_to_image_tensor(frames)
        frame_count = int(images.shape[0])
        out_info = build_video_info(
            video_info,
            frame_count=frame_count,
            frame_rate=float(frame_rate),
            width=int(images.shape[2]),
            height=int(images.shape[1]),
        )
        return (images, frame_count, float(frame_rate), audio, out_info)
