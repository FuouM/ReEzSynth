# ezsynth/utils/warp_utils.py
import cv2
import numpy as np


class Warp:
    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width
        self.grid = self._create_grid(self.H, self.W)

    def _create_grid(self, H: int, W: int):
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        return np.stack((x, y), axis=-1).astype(np.float32)

    def _warp(self, img: np.ndarray, flo: np.ndarray):
        # The input image for warping must be float32
        flo_resized = cv2.resize(flo, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]

        warped_img = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped_img

    def run_warping(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warps an image using an optical flow field.
        Handles both uint8 color images and float32 data maps (like guides) correctly.
        """
        was_uint8 = img.dtype == np.uint8

        # CRITICAL FIX: Only normalize if the input is a uint8 color image.
        # Float images (like coordinate maps) must NOT be normalized.
        if was_uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32)

        warped_float = self._warp(img_float, flow.astype(np.float32))

        # Return to original type
        if was_uint8:
            return (warped_float * 255).clip(0, 255).astype(np.uint8)
        else:
            return warped_float


class PositionalGuide:
    """A stateless factory for creating positional guides."""

    def __init__(self, height: int, width: int):
        self.warp = Warp(height, width)
        self.pristine_coord_map = self._create_coord_map(height, width)

    def _create_coord_map(self, H: int, W: int):
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)
        # The coordinate map is float32 data in the [0, 1] range.
        return np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.float32)

    def get_pristine_guide_uint8(self) -> np.ndarray:
        """Returns the pristine guide as a uint8 image for ebsynth."""
        return (self.pristine_coord_map * 255).astype(np.uint8)

    def create_from_flow(self, flow: np.ndarray) -> np.ndarray:
        """
        Creates a new target positional guide by warping the pristine map.
        Returns a uint8 image for ebsynth.
        """
        # Warping the float32 pristine map is correct.
        coord_map_warped = self.warp.run_warping(self.pristine_coord_map, flow)

        # Apply modulo to wrap coordinates, preventing tiling from out-of-bounds values
        coord_map_warped[..., :2] = coord_map_warped[..., :2] % 1.0

        return (coord_map_warped * 255).clip(0, 255).astype(np.uint8)
