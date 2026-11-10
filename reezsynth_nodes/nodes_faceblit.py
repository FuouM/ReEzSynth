"""ComfyUI node for FaceBlit (landmark-guided example-based face stylization)."""

from __future__ import annotations

from pathlib import Path

import torch

from .image_io import (
    bgr_uint8_sequence_to_image_tensor,
    image_batch_to_bgr_uint8_list,
    image_tensor_to_bgr_uint8,
)
from .nodes_image_synth import _bool, _combo, _int
from .options import (
    FACEBLIT_BACKENDS,
    FACEBLIT_DFS_MODES,
    FACEBLIT_LANDMARK_DETECTORS,
    parse_landmarks,
    require_module,
)

_LANDMARKS_TOOLTIP = (
    "Optional 68-point landmarks of the {what}: 68 lines 'x y' (FaceBlit .txt layout, "
    "optional leading '68' line), a flat 'x1,y1,x2,y2,...' list, or a path to such a "
    ".txt file. Empty: detect automatically (needs `pip install face-alignment`)."
)


def landmarks_from_input(text: str, name: str) -> list[tuple[int, int]] | None:
    """Landmarks from inline text or from a FaceBlit landmark file path."""
    stripped = text.strip()
    if stripped and "\n" not in stripped and stripped.lower().endswith(".txt"):
        path = Path(stripped.strip('"'))
        if not path.is_file():
            raise FileNotFoundError(f"{name}: landmark file not found: {path}")
        stripped = path.read_text()
    return parse_landmarks(stripped, name)


class ReEzSynthFaceBlit:
    DESCRIPTION = (
        "FaceBlit: transfer a painted/stylized face onto target face photos using 68 "
        "facial landmarks (MLS-warped positional guide + appearance guide). Works on a "
        "single target or a batch of frames; style guides are computed once."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": (
                    "IMAGE",
                    {"tooltip": "Stylized face exemplar (one image)."},
                ),
                "target_image": (
                    "IMAGE",
                    {
                        "tooltip": "Face photo(s) to stylize; a batch is processed frame by frame."
                    },
                ),
                "backend": _combo(
                    FACEBLIT_BACKENDS,
                    "torch",
                    "torch: no extra install. taichi: needs `pip install taichi`.",
                ),
                "landmark_detector": _combo(
                    FACEBLIT_LANDMARK_DETECTORS,
                    "fan_blazeface",
                    "face-alignment FAN with the BlazeFace (small, fast) or S3FD (more "
                    "robust, larger download) face detector. Unused when landmarks are given.",
                ),
                "stylize_background": _bool(
                    False,
                    "Stylize the whole frame; off blends the stylized face into the photo.",
                ),
                "use_vectorized": _bool(
                    True,
                    "LUT lookup NNF (fast, GPU). Off: FaceBlit's seed-growing DFS on the "
                    "CPU (closer to the original, slower; see dfs_mode).",
                ),
                "dfs_mode": _combo(
                    FACEBLIT_DFS_MODES,
                    "numba",
                    "CPU DFS implementation when use_vectorized is off (torch backend): "
                    "numba (fast, needs numba) or python (slow, no extra deps).",
                ),
                "grid_size": _int(3, 1, 16, 1, "MLS warp grid spacing in pixels."),
                "patch_size": _int(3, 1, 15, 2, "Voting patch size (odd)."),
                "lambda_pos": _int(
                    10, 0, 100, 1, "Weight of the positional guide in matching."
                ),
                "lambda_app": _int(
                    2, 0, 100, 1, "Weight of the appearance guide in matching."
                ),
                "blend_sigma": _int(
                    25, 1, 100, 1, "Feathering of the face mask when compositing."
                ),
            },
            "optional": {
                "style_landmarks": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": _LANDMARKS_TOOLTIP.format(what="style image"),
                    },
                ),
                "target_landmarks": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": _LANDMARKS_TOOLTIP.format(
                            what="target image (single image only)"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "target_pos_guide")
    OUTPUT_TOOLTIPS = (
        "Stylized target image(s).",
        "Positional guide warped to the target face (diagnostic).",
    )
    FUNCTION = "stylize"
    CATEGORY = "ReEzSynth/face"

    def stylize(
        self,
        style_image,
        target_image,
        backend,
        landmark_detector,
        stylize_background,
        use_vectorized,
        dfs_mode,
        grid_size,
        patch_size,
        lambda_pos,
        lambda_app,
        blend_sigma,
        style_landmarks="",
        target_landmarks="",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}.")
        if backend == "taichi":
            require_module("taichi")
        style_lm = landmarks_from_input(style_landmarks, "style_landmarks")
        target_lm = landmarks_from_input(target_landmarks, "target_landmarks")
        style_bgr = image_tensor_to_bgr_uint8(style_image)
        targets = image_batch_to_bgr_uint8_list(target_image)
        if target_lm is not None and len(targets) != 1:
            raise ValueError(
                "target_landmarks describe one face; clear it to detect landmarks per "
                f"frame of the {len(targets)}-frame target batch."
            )
        if style_lm is None or target_lm is None:
            require_module("face_alignment")
        if not use_vectorized and backend == "torch" and dfs_mode == "numba":
            require_module("numba")

        from FaceBlit.src.engine import (
            FaceBlitEngine,
            FaceBlitInput_Style,
            FaceBlitInput_Target,
        )
        from FaceBlit.src.utils_face import detect_landmarks

        from . import comfy_runtime as rt

        device = rt.torch_device()
        device_name = str(device)
        detector = landmark_detector.removeprefix("fan_")

        def detect(image, what: str):
            found = detect_landmarks(image, "fan", None, device_name, detector)
            if found is None:
                raise ValueError(
                    f"No face detected in {what}. Try landmark_detector=fan_sfd or "
                    "supply the landmarks manually."
                )
            return found

        # The style LUT search keeps a 256^3 int32 table plus guides on the device.
        rt.prepare_vram(device, 1 << 30)
        engine = FaceBlitEngine()
        results = []
        pos_guides = []
        try:
            with rt.comfy_progress() as report:
                total = len(targets) + 1
                style = FaceBlitInput_Style.from_image(
                    style_bgr,
                    style_lm
                    if style_lm is not None
                    else detect(style_bgr, "style_image"),
                    device_name,
                    lambda_pos=int(lambda_pos),
                    lambda_app=int(lambda_app),
                    backend=backend,
                )
                report("faceblit", 1, total)
                for i, target_bgr in enumerate(targets):
                    landmarks = (
                        target_lm
                        if target_lm is not None
                        else detect(target_bgr, f"target_image frame {i}")
                    )
                    result, pos_guide = engine.stylize_image_with_guide_and_landmarks(
                        input_target=FaceBlitInput_Target(target_bgr, landmarks),
                        input_style=style,
                        device=device_name,
                        dfs_mode=dfs_mode,
                        grid_size=int(grid_size),
                        patch_size=int(patch_size),
                        lambda_pos=int(lambda_pos),
                        lambda_app=int(lambda_app),
                        use_vectorized=bool(use_vectorized),
                        blend_sigma=int(blend_sigma),
                        stylize_background=bool(stylize_background),
                        backend=backend,
                    )
                    results.append(result)
                    pos_guides.append(pos_guide)
                    report("faceblit", i + 2, total)
        finally:
            rt.release_vram()

        return (
            bgr_uint8_sequence_to_image_tensor(results),
            bgr_uint8_sequence_to_image_tensor(pos_guides),
        )
