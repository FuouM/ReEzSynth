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
        """Warp an image using forward splatting, falling back to regular warping."""
        if not (
            self.use_taichi
            and self._taichi_available
            and hasattr(self.ops, "soft_splat_kernel")
        ):
            del fill_holes, src_guide, tgt_guide
            if return_weight:
                return (
                    self.run_warping(img, flow),
                    np.ones((self.H, self.W), dtype=np.float32),
                )
            return self.run_warping(img, flow)

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
        flow_f32 = _as_float32_array(flow)

        dst_color = np.zeros_like(img_float)
        dst_weight = np.zeros((self.H, self.W), dtype=np.float32)
        use_bilateral = src_guide is not None and tgt_guide is not None
        src_guide_f32 = (
            _as_float32_array(src_guide)
            if use_bilateral
            else np.zeros((1, 1, 3), dtype=np.float32)
        )
        tgt_guide_f32 = (
            _as_float32_array(tgt_guide)
            if use_bilateral
            else np.zeros((1, 1, 3), dtype=np.float32)
        )

        self.ops.soft_splat_kernel(
            img_float,
            flow_f32,
            dst_color,
            dst_weight,
            src_guide_f32,
            tgt_guide_f32,
            use_bilateral,
        )

        raw_weight = dst_weight.copy()
        if fill_holes:
            self.run_pull_push(dst_color, dst_weight)

        out = np.zeros_like(img, dtype=np.uint8 if was_uint8 else np.float32)
        self.ops.normalize_splat_kernel(dst_color, dst_weight, out, was_uint8)
        if return_weight:
            return out, raw_weight
        return out

    def run_pull_push(self, color: np.ndarray, weight: np.ndarray, levels: int = 5) -> None:
        """Hierarchical hole filling using pull-push accumulation."""
        pyramid_color = [color]
        pyramid_weight = [weight]

        for _ in range(levels - 1):
            h, w = pyramid_color[-1].shape[:2]
            if h <= 2 or w <= 2:
                break
            h_next, w_next = h // 2, w // 2
            color_shape = pyramid_color[-1].shape[2:]
            next_color = np.zeros((h_next, w_next, *color_shape), dtype=np.float32)
            next_weight = np.zeros((h_next, w_next), dtype=np.float32)
            self.ops.pull_kernel(
                pyramid_color[-1],
                pyramid_weight[-1],
                next_color,
                next_weight,
            )
            pyramid_color.append(next_color)
            pyramid_weight.append(next_weight)

        for i in range(len(pyramid_color) - 2, -1, -1):
            self.ops.push_kernel(
                pyramid_color[i + 1],
                pyramid_weight[i + 1],
                pyramid_color[i],
                pyramid_weight[i],
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
