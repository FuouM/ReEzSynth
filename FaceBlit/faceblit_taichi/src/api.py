import numpy as np
import torch

from .backend import FaceBlitTaichiBackend


class FaceBlitTaichi:
    def __init__(self):
        self.backend = FaceBlitTaichiBackend()
        self.style_image = None
        self.style_pos_guide = None
        self.style_app_guide = None
        self.look_up_cube = None

    def load_style(
        self, style_image, style_pos_guide, style_app_guide, lut_packed=None
    ):
        """
        Expects images as torch tensors or numpy arrays.
        """
        self.style_image = self._to_tensor(style_image)
        self.style_pos_guide = self._to_tensor(style_pos_guide)
        self.style_app_guide = self._to_tensor(style_app_guide)

        if lut_packed is not None:
            self.look_up_cube = self._to_tensor(lut_packed)
        else:
            print("[FaceBlit Taichi] Computing LUT...")
            self.look_up_cube = self.backend.compute_lut(
                self.style_pos_guide, self.style_app_guide
            )

    def stylize_with_guides(self, target_pos_guide, target_app_guide, patch_size=3):
        if self.look_up_cube is None:
            raise ValueError("Style or LUT not loaded. Call load_style first.")

        target_pos_guide = self._to_tensor(target_pos_guide)
        target_app_guide = self._to_tensor(target_app_guide)

        return self.backend.stylize(
            self.style_image,
            target_pos_guide,
            target_app_guide,
            self.look_up_cube,
            patch_size=patch_size,
        )

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
