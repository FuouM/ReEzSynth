# ezsynth/utils/warp_utils.py
import cv2
import numpy as np


class Warp:
    def __init__(self, height: int, width: int, use_taichi: bool = False):
        self.H = height
        self.W = width
        self.use_taichi = use_taichi
        self.grid = self._create_grid(self.H, self.W)
        self._taichi_available = False

        if self.use_taichi:
            try:
                from ..engines.backends.taichi_backend import ensure_ti_init
                from ..engines.backends.taichi_ops import TaichiOps

                ensure_ti_init()
                self.ops = TaichiOps()
                self._taichi_available = True
            except ImportError:
                self.use_taichi = False

    def _create_grid(self, H: int, W: int):
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        return np.stack((x, y), axis=-1).astype(np.float32)

    def _warp(self, img: np.ndarray, flo: np.ndarray, interpolation=cv2.INTER_LINEAR):
        if flo.shape[:2] == (self.H, self.W):
            flo_resized = flo
        else:
            flo_resized = cv2.resize(flo, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        if self.use_taichi and self._taichi_available and interpolation == cv2.INTER_LINEAR:
            dst = np.zeros_like(img)
            self.ops.bilinear_warp_kernel(img, flo_resized, dst)
            return dst

        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]

        warped_img = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped_img

    def run_warping(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warps an image using an optical flow field.
        Handles both uint8 color images and float32 data maps (like guides) correctly.
        """
        if _is_identity_flow(flow, self.H, self.W):
            return img

        was_uint8 = img.dtype == np.uint8

        if was_uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = _as_float32_array(img)

        warped_float = self._warp(img_float, _as_float32_array(flow))

        if was_uint8:
            return (warped_float * 255).clip(0, 255).astype(np.uint8)
        return warped_float

    def run_warping_float_map(
        self, img: np.ndarray, flow: np.ndarray, interpolation=cv2.INTER_LINEAR
    ) -> np.ndarray:
        """Warp a numeric map and return float32 without uint8 normalization."""
        if _is_identity_flow(flow, self.H, self.W):
            return _as_float32_array(img).copy()
        return self._warp(
            _as_float32_array(img),
            _as_float32_array(flow),
            interpolation=interpolation,
        )

    def run_forward_warping(
        self,
        img: np.ndarray,
        flow: np.ndarray,
        fill_holes: bool = True,
        return_weight=False,
        src_guide: np.ndarray = None,
        tgt_guide: np.ndarray = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Warp an image using forward splatting when available.

        The current Taichi backend does not expose splat kernels yet, so this
        falls back to regular warping while preserving the refactor API.
        """
        del fill_holes, src_guide, tgt_guide
        if return_weight:
            return (
                self.run_warping(img, flow),
                np.ones((self.H, self.W), dtype=np.float32),
            )
        return self.run_warping(img, flow)


def _as_float32_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32:
        return np.ascontiguousarray(array) if not array.flags.c_contiguous else array
    return array.astype(np.float32)


def _is_identity_flow(flow: np.ndarray, height: int, width: int) -> bool:
    return flow.shape[:2] == (height, width) and not np.any(flow)


class PositionalGuide:
    """A stateless factory for creating positional guides."""

    def __init__(
        self,
        height: int,
        width: int,
        use_taichi: bool = False,
        use_forward_warp: bool = False,
    ):
        self.warp = Warp(height, width, use_taichi=use_taichi)
        self.use_forward_warp = use_forward_warp
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
        if self.use_forward_warp:
            coord_map_warped = self.warp.run_forward_warping(
                self.pristine_coord_map,
                flow,
            )
        else:
            coord_map_warped = self.warp.run_warping(self.pristine_coord_map, flow)

        # Apply modulo to wrap coordinates, preventing tiling from out-of-bounds values
        coord_map_warped[..., :2] = coord_map_warped[..., :2] % 1.0

        return (coord_map_warped * 255).clip(0, 255).astype(np.uint8)
