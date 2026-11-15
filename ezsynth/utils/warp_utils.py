# ezsynth/utils/warp_utils.py
import cv2
import numpy as np

from ..kernels.imageops import ImageOpsEngine, load


class Warp:
    """Flow warping on OpenCV, Taichi or WebGPU (``engine``, see ``kernels.imageops``).

    Backward warping works on every engine; forward splatting and pull-push hole
    filling need ``taichi`` or ``wgpu``.
    """

    def __init__(self, height: int, width: int, engine: ImageOpsEngine = "opencv"):
        self.H = height
        self.W = width
        self.engine = engine
        self.ops = load(engine)

    @property
    def can_splat(self) -> bool:
        """Forward splatting and pull-push are available."""
        return "forward_splat" in self.ops.OPS

    def _warp(self, img: np.ndarray, flo: np.ndarray, interpolation=cv2.INTER_LINEAR):
        if interpolation not in (cv2.INTER_LINEAR, cv2.INTER_NEAREST):
            raise ValueError("Warping takes INTER_LINEAR or INTER_NEAREST only.")
        if flo.shape[:2] == (self.H, self.W):
            flo_resized = flo
        else:
            flo_resized = cv2.resize(
                flo, (self.W, self.H), interpolation=cv2.INTER_LINEAR
            )
        return self.ops.warp(
            img, flo_resized, nearest=interpolation == cv2.INTER_NEAREST
        )

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
        """Warp an image by forward soft splatting (``taichi`` / ``wgpu`` engines)."""
        if not self.can_splat:
            raise ValueError(
                "Forward splatting needs the 'taichi' or 'wgpu' image ops engine."
            )

        if _is_identity_flow(flow, self.H, self.W):
            out = (
                _as_uint8_array(img).copy()
                if img.dtype == np.uint8
                else _as_float32_array(img).copy()
            )
            if return_weight:
                return out, np.ones((self.H, self.W), dtype=np.float32)
            return out

        was_uint8 = img.dtype == np.uint8
        img_float = img.astype(np.float32) if was_uint8 else _as_float32_array(img)
        use_bilateral = src_guide is not None and tgt_guide is not None
        out, raw_weight = self.ops.forward_splat(
            img_float,
            _as_float32_array(flow),
            fill_holes=fill_holes,
            src_guide=_as_float32_array(src_guide) if use_bilateral else None,
            tgt_guide=_as_float32_array(tgt_guide) if use_bilateral else None,
            uint8_out=was_uint8,
        )
        if return_weight:
            return out, raw_weight
        return out

    def run_pull_push(
        self, color: np.ndarray, weight: np.ndarray, levels: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hierarchical hole filling of accumulated ``color`` / ``weight``.

        Returns the filled ``(color, weight)``; the Taichi engine fills the given
        arrays in place.
        """
        if not self.can_splat:
            raise ValueError("Pull-push needs the 'taichi' or 'wgpu' image ops engine.")
        return self.ops.pull_push(
            _as_float32_array(color), _as_float32_array(weight), levels=levels
        )


def _as_uint8_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array) if not array.flags.c_contiguous else array
    safe = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return safe.clip(0, 255).astype(np.uint8)


def _as_float32_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32:
        out = np.ascontiguousarray(array) if not array.flags.c_contiguous else array
    else:
        out = array.astype(np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _is_identity_flow(flow: np.ndarray, height: int, width: int) -> bool:
    return flow.shape[:2] == (height, width) and not np.any(flow)


class PositionalGuide:
    """A stateless factory for creating positional guides."""

    def __init__(
        self,
        height: int,
        width: int,
        engine: ImageOpsEngine = "opencv",
        use_forward_warp: bool = False,
    ):
        self.warp = Warp(height, width, engine=engine)
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
