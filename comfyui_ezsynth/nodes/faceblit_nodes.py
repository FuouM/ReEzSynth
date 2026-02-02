# comfyui_ezsynth/nodes/faceblit_nodes.py
"""
FaceBlit nodes for face-aware stylization.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .base import EZBaseNode


class FaceBlitStyleNode(EZBaseNode):
    """
    Precompute style assets for FaceBlit face-aware stylization.
    """

    CATEGORY = "ReEzSynth/FaceBlit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
            },
            "optional": {
                "landmark_model": (["dlib", "fan"], {"default": "dlib"}),
                "predictor_path": ("STRING", {"default": ""}),
                "stretch_hist": ("BOOLEAN", {"default": True}),
                "lambda_pos": ("INT", {"default": 10}),
                "lambda_app": ("INT", {"default": 2}),
            },
        }

    RETURN_TYPES = ("FB_ASSETS",)
    RETURN_NAMES = ("style_assets",)
    FUNCTION = "compute_style_assets"

    def compute_style_assets(
        self,
        style_image: torch.Tensor,
        landmark_model: str = "dlib",
        predictor_path: str = "",
        stretch_hist: bool = True,
        lambda_pos: int = 10,
        lambda_app: int = 2,
    ) -> Tuple[Dict]:
        """
        Compute FaceBlit style assets.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        from ..core.tensor_utils import tensor_to_numpy

        style_np = tensor_to_numpy(style_image)
        temp_dir = Path(tempfile.mkdtemp())
        style_path = temp_dir / "style.png"

        try:
            import cv2

            cv2.imwrite(str(style_path), style_np)
        except:
            from PIL import Image

            Image.fromarray(style_np).save(style_path)

        try:
            from FaceBlit.faceblit_pytorch.src.api import compute_style_assets

            assets = compute_style_assets(
                input_path=style_path,
                output_dir=temp_dir,
                predictor_path=predictor_path if predictor_path else None,
                landmark_model=landmark_model,
                stretch_hist=stretch_hist,
                lambda_pos=lambda_pos,
                lambda_app=lambda_app,
                reuse_precomputed=False,
            )

            return (assets,)

        except ImportError:
            raise ImportError("FaceBlit is not properly installed.")


class FaceDetectNode(EZBaseNode):
    """
    Detect facial landmarks in an image.
    """

    CATEGORY = "ReEzSynth/FaceBlit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model": (["dlib", "fan"], {"default": "dlib"}),
                "predictor_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LANDMARKS", "BOOLEAN")
    RETURN_NAMES = ("landmarks", "face_found")
    FUNCTION = "detect_face"

    def detect_face(
        self,
        image: torch.Tensor,
        model: str = "dlib",
        predictor_path: str = "",
    ) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Detect facial landmarks.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        from ..core.tensor_utils import tensor_to_numpy

        image_np = tensor_to_numpy(image)
        temp_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            import cv2

            cv2.imwrite(str(temp_path), image_np)
        except:
            from PIL import Image

            Image.fromarray(image_np).save(temp_path)

        try:
            import dlib

            if model == "dlib":
                detector = dlib.get_frontal_face_detector()
                sp = dlib.shape_predictor(predictor_path if predictor_path else "")
                img = cv2.imread(str(temp_path))
                dets = detector(img, 1)
                if len(dets) == 0:
                    return ([], False)

                shape = sp(img, dets[0])
                landmarks = [
                    (shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)
                ]

            else:
                from FaceBlit.faceblit_pytorch.src.api import _get_fan_model

                img_rgb = image_np[..., ::-1].copy()
                fa = _get_fan_model("cpu")
                preds = fa.get_landmarks(img_rgb)

                if preds is None or len(preds) == 0:
                    return ([], False)

                landmarks = [(int(p[0]), int(p[1])) for p in preds[0]]

            return (landmarks, True)

        except ImportError:
            raise ImportError("Face detection dependencies not installed.")


class FaceBlitGuideNode(EZBaseNode):
    """
    Generate position and appearance guides for FaceBlit.
    """

    CATEGORY = "ReEzSynth/FaceBlit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "stretch_hist": ("BOOLEAN", {"default": True}),
                "draw_grid": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pos_guide", "app_guide")
    FUNCTION = "generate_guides"

    def generate_guides(
        self,
        image: torch.Tensor,
        stretch_hist: bool = True,
        draw_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FaceBlit guides.
        """
        from FaceBlit.faceblit_pytorch.src.api import get_app_guide, get_gradient

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        image_np = tensor_to_numpy(image)
        h, w = image_np.shape[:2]

        pos_guide_np = get_gradient(w, h, draw_grid=draw_grid)
        app_guide_np = get_app_guide(image_np, stretch_hist=stretch_hist)

        return (
            self._format_output(numpy_to_tensor(pos_guide_np)),
            self._format_output(numpy_to_tensor(app_guide_np)),
        )


class FaceBlitStylizeNode(EZBaseNode):
    """
    Apply face-aware stylization using FaceBlit.
    """

    CATEGORY = "ReEzSynth/FaceBlit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "style_assets": ("FB_ASSETS",),
                "target_landmarks": ("LANDMARKS",),
                "target_app_guide": ("IMAGE",),
            },
            "optional": {
                "stylize_bg": ("BOOLEAN", {"default": False}),
                "patch_size": ("INT", {"default": 3, "min": 1, "max": 7}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stylized", "target_pos_guide")
    FUNCTION = "stylize"

    def stylize(
        self,
        target_image: torch.Tensor,
        style_assets: Dict,
        target_landmarks: List[Tuple[int, int]],
        target_app_guide: torch.Tensor,
        stylize_bg: bool = False,
        patch_size: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FaceBlit stylization.
        """
        from FaceBlit.faceblit_pytorch.src.api import FaceBlit

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        target_np = tensor_to_numpy(target_image)
        app_guide_np = tensor_to_numpy(target_app_guide)

        faceblit = FaceBlit()
        faceblit.load_style_with_guides(
            style_path=style_assets["style_path"],
            landmarks_path=style_assets["landmarks_path"],
            lut_path=style_assets["lut_path"],
            style_pos_guide_path=style_assets["pos_guide_path"],
            style_app_guide_path=style_assets["app_guide_path"],
        )

        result, target_pos_guide = faceblit.stylize_image_with_guide_and_landmarks(
            image_bgr=target_np,
            target_app_guide=app_guide_np,
            target_landmarks=target_landmarks,
            stylize_bg=stylize_bg,
            patch_size=patch_size,
        )

        return (
            self._format_output(numpy_to_tensor(result)),
            self._format_output(numpy_to_tensor(target_pos_guide)),
        )
