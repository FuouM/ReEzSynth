"""ComfyUI node for FastBlend keyframe interpolation."""

from __future__ import annotations

import torch

from .image_io import bgr_uint8_sequence_to_image_tensor, image_batch_to_bgr_uint8_list
from .nodes_image_synth import _combo, _float, _int
from .options import (
    FASTBLEND_ACCURACY,
    FASTBLEND_BACKENDS,
    check_indices_in_range,
    has_module,
    parse_indices,
    require_module,
)

FASTBLEND_BACKEND_HINT = (
    "FastBlend needs one patch-matching backend: `pip install taichi` (recommended), "
    "`pip install cupy-cuda12x`, or the native 'cuda' extension (CUDA toolkit + C++ "
    "compiler for the JIT build)."
)


def resolve_backend(requested: str) -> str:
    """Concrete FastBlend backend. ``auto`` prefers installed Python packages
    (taichi, then cupy) so ComfyUI never starts a surprise C++/CUDA JIT build;
    pick ``cuda`` explicitly for the native extension."""
    if requested == "auto":
        for module in ("taichi", "cupy"):
            if has_module(module):
                return module
        raise ImportError(
            "ReEzSynth FastBlend: no Python backend installed. "
            + FASTBLEND_BACKEND_HINT
        )
    if requested in ("taichi", "cupy"):
        require_module(requested)
    return requested


class ReEzSynthFastBlend:
    DESCRIPTION = (
        "FastBlend keyframe interpolation: propagates stylized keyframes to every "
        "content frame by patch matching between neighbouring frames (fast, no flow)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_images": (
                    "IMAGE",
                    {"tooltip": "All content (guide) frames as one IMAGE batch."},
                ),
                "keyframe_images": (
                    "IMAGE",
                    {
                        "tooltip": "Stylized keyframes, in the order of keyframe_indices."
                    },
                ),
                "keyframe_indices": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "Content frame index of each keyframe, e.g. '0, 10'.",
                    },
                ),
                "backend": _combo(
                    FASTBLEND_BACKENDS,
                    "auto",
                    "auto: taichi, else cupy (installed packages only). cuda: native "
                    "extension (JIT build needs CUDA toolkit + compiler).",
                ),
                "accuracy": _combo(
                    tuple(FASTBLEND_ACCURACY),
                    "balanced",
                    "Patch matching effort: fast, balanced or accurate.",
                ),
                "window_size": _int(
                    5, 1, 25, 1, "Neighbouring frames blended on each side."
                ),
                "batch_size": _int(
                    8, 1, 128, 1, "Frames matched per GPU batch (lower uses less VRAM)."
                ),
                "minimum_patch_size": _int(
                    15, 3, 31, 2, "Patch size at the finest pyramid level (odd)."
                ),
                "num_iter": _int(
                    5, 1, 20, 1, "PatchMatch iterations per pyramid level."
                ),
                "guide_weight": _float(
                    10.0,
                    0.0,
                    100.0,
                    0.1,
                    "Weight of the content guide vs. the style in matching.",
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Stylized frames, one per content frame.",)
    FUNCTION = "interpolate"
    CATEGORY = "ReEzSynth/video"

    def interpolate(
        self,
        content_images,
        keyframe_images,
        keyframe_indices,
        backend,
        accuracy,
        window_size,
        batch_size,
        minimum_patch_size,
        num_iter,
        guide_weight,
    ) -> tuple[torch.Tensor]:
        if minimum_patch_size % 2 == 0:
            raise ValueError(
                f"minimum_patch_size must be odd, got {minimum_patch_size}."
            )
        indices = parse_indices(keyframe_indices, "keyframe_indices")
        guide_frames = image_batch_to_bgr_uint8_list(content_images)
        style_frames = image_batch_to_bgr_uint8_list(keyframe_images)
        if len(style_frames) != len(indices):
            raise ValueError(
                f"Got {len(style_frames)} keyframe images but {len(indices)} "
                "keyframe_indices; give one index per keyframe."
            )
        check_indices_in_range(indices, len(guide_frames), "keyframe_indices")
        resolved = resolve_backend(backend)

        from ezsynth.utils.fastblend_cli import (
            create_interpolation_config,
            interpolate_frames,
        )
        from ezsynth.utils.image_utils import resize_image_to_match

        from . import comfy_runtime as rt

        order = sorted(range(len(indices)), key=indices.__getitem__)
        indices = [indices[i] for i in order]
        style_frames = [
            resize_image_to_match(style_frames[i], guide_frames[0]) for i in order
        ]

        config = create_interpolation_config(
            {
                "accuracy": FASTBLEND_ACCURACY[accuracy],
                "window_size": int(window_size),
                "batch_size": int(batch_size),
                "minimum_patch_size": int(minimum_patch_size),
                "num_iter": int(num_iter),
                "guide_weight": float(guide_weight),
                "backend": resolved,
            }
        )
        device = rt.torch_device()
        height, width = guide_frames[0].shape[:2]
        # Pyramid matcher: guide+style batches, NNF and error buffers per batch frame.
        rt.prepare_vram(device, (512 << 20) + height * width * int(batch_size) * 96)
        try:
            with rt.comfy_progress() as report:
                frames = interpolate_frames(
                    guide_frames,
                    style_frames,
                    indices,
                    config,
                    progress_callback=lambda done, total: report(
                        "fastblend", done, total
                    ),
                )
        finally:
            rt.release_vram()
        return (bgr_uint8_sequence_to_image_tensor(frames),)
