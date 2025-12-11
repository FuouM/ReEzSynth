import datetime
from pathlib import Path
from typing import Sequence, Tuple, Union

import dlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import ops

PathLike = Union[str, Path]

# Global cache for FAN model to avoid reloading
_fan_model_cache = {}


def get_timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _get_fan_model(device: str, model: str = "dlib"):
    """Get cached FAN model or create new one for the specified device."""
    if device in _fan_model_cache:
        print(f"[{get_timestamp()}] [FAN] Using cached model for device: {device}")
        return _fan_model_cache[device]

    print(f"[{get_timestamp()}] [FAN] Loading model for device: {device}...")
    import time

    start_time = time.time()

    try:
        import face_alignment
    except ImportError:
        raise ImportError(
            "face-alignment package is not installed. Please install it with `pip install face-alignment`."
        )

    # Determine landmarks type with version compatibility
    landmarks_type = getattr(face_alignment.LandmarksType, "_2D", None)
    if landmarks_type is None:
        landmarks_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
    if landmarks_type is None:
        raise AttributeError(
            "face_alignment.LandmarksType is missing _2D/TWO_D; "
            "please install a compatible face-alignment version."
        )

    fa = face_alignment.FaceAlignment(
        landmarks_type, device=device, flip_input=True, face_detector=model
    )

    load_time = time.time() - start_time
    print(
        f"[{get_timestamp()}] [FAN] Model loaded for device: {device} with {model} in {load_time:.3f}s"
    )
    _fan_model_cache[device] = fa
    return fa


def _to_path(path: PathLike) -> str:
    return str(Path(path))


def _read_image_pil(path: PathLike) -> np.ndarray:
    """Read image using PIL and return as BGR numpy array (OpenCV format)."""
    if Image is None:
        raise RuntimeError(
            "PIL is required for image reading when cv2 is not available."
        )
    img = Image.open(_to_path(path)).convert("RGB")
    arr = np.array(img)
    # Convert RGB to BGR for consistency with OpenCV
    return arr[..., ::-1].copy()


def _write_image_pil(path: PathLike, image: np.ndarray) -> None:
    """Write BGR numpy array as image using PIL."""
    if Image is None:
        raise RuntimeError(
            "PIL is required for image writing when cv2 is not available."
        )
    # Convert BGR to RGB
    rgb = image[..., ::-1]
    img = Image.fromarray(rgb)
    img.save(_to_path(path))


def _resize_torch(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image using PyTorch. size is (width, height)."""
    # Convert to tensor (H, W, C) -> (1, C, H, W)
    if image.ndim == 2:
        t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        t_resized = F.interpolate(
            t, size=(size[1], size[0]), mode="bilinear", align_corners=False
        )
        return t_resized.squeeze(0).squeeze(0).numpy().astype(image.dtype)
    else:
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        t_resized = F.interpolate(
            t, size=(size[1], size[0]), mode="bilinear", align_corners=False
        )
        return t_resized.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)


def _normalize_landmarks(landmarks: Sequence[Tuple[int, int]]) -> list[Tuple[int, int]]:
    return [(int(x), int(y)) for x, y in landmarks]


class FaceBlit:
    """Pure Python/PyTorch implementation of the FaceBlit engine."""

    def __init__(
        self,
        model_path: PathLike | None = None,
        device: str | torch.device | None = None,
        precompute_device: str | torch.device | None = None,
        stylize_device: str | torch.device | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else None

        # Handle device assignment for backward compatibility and new functionality
        if device is not None:
            # If device is specified, use it for both operations (backward compatibility)
            device_obj = torch.device(device)
            self.precompute_device = device_obj
            self.stylize_device = device_obj
        else:
            # Use separate devices if specified, otherwise default to CPU
            self.precompute_device = (
                torch.device(precompute_device)
                if precompute_device is not None
                else torch.device("cpu")
            )
            self.stylize_device = (
                torch.device(stylize_device)
                if stylize_device is not None
                else torch.device("cpu")
            )

        self.style_image: np.ndarray | None = None
        self.style_landmarks: list[Tuple[int, int]] | None = None
        self.style_pos_guide: np.ndarray | None = None
        self.style_app_guide: np.ndarray | None = None
        self.look_up_cube: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Loading / setup
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.style_image is None or self.style_landmarks is None:
            raise RuntimeError(
                "Style not loaded. Call load_style or load_style_with_guides first."
            )

    def load_style(
        self, style_path: PathLike, landmarks_path: PathLike, lut_path: PathLike
    ) -> None:
        """Load style using PIL/PyTorch for image I/O."""
        style_img = _read_image_pil(style_path)
        if style_img is None or style_img.size == 0:
            raise FileNotFoundError(f"Failed to read style image: {style_path}")
        self.style_image = style_img
        self.style_landmarks = ops.read_landmarks_file(landmarks_path)
        self.style_pos_guide = ops.gradient_guide(
            style_img.shape[1], style_img.shape[0], draw_grid=False
        )
        self.style_app_guide = ops.get_app_guide(style_img, stretch_hist=True)
        self.look_up_cube = ops.load_look_up_cube(_to_path(lut_path))

    def load_style_with_guides(
        self,
        style_path: PathLike,
        landmarks_path: PathLike,
        lut_path: PathLike,
        style_pos_guide_path: PathLike,
        style_app_guide_path: PathLike,
    ) -> None:
        """Load style with precomputed guides using PIL/PyTorch."""
        style_img = _read_image_pil(style_path)
        if style_img is None or style_img.size == 0:
            raise FileNotFoundError(f"Failed to read style image: {style_path}")
        self.style_image = style_img
        self.style_landmarks = ops.read_landmarks_file(landmarks_path)
        pos = _read_image_pil(style_pos_guide_path)
        # Read grayscale guide
        if Image is not None:
            app_img = Image.open(_to_path(style_app_guide_path)).convert("L")
            app = np.array(app_img)
        else:
            raise RuntimeError("PIL is required to read images.")
        if pos is None or app is None:
            raise FileNotFoundError("Failed to read provided guides.")
        self.style_pos_guide = pos
        self.style_app_guide = app
        self.look_up_cube = ops.load_look_up_cube(_to_path(lut_path))

    # ------------------------------------------------------------------
    # Stylization
    # ------------------------------------------------------------------
    def stylize_image_with_guide_and_landmarks(
        self,
        image_bgr: np.ndarray,
        target_app_guide: np.ndarray,
        target_landmarks: Sequence[Tuple[int, int]],
        *,
        stylize_bg: bool = False,
        patch_size: int = 3,
        use_vectorized: bool = False,
        dfs_mode: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        assert self.style_image is not None
        assert self.style_pos_guide is not None
        assert self.style_app_guide is not None
        assert self.style_landmarks is not None

        target_img = ops._to_uint8(image_bgr)
        tgt_landmarks = ops.clamp_landmarks(
            _normalize_landmarks(target_landmarks), target_img.shape[:2]
        )
        style_landmarks = ops.clamp_landmarks(
            self.style_landmarks, self.style_image.shape[:2]
        )

        # Match C++: when stylizing background, anchor MLS with bottom corners
        mls_style_landmarks = list(style_landmarks)
        mls_target_landmarks = list(tgt_landmarks)
        if stylize_bg:
            style_h, style_w = self.style_image.shape[:2]
            tgt_h, tgt_w = target_img.shape[:2]
            mls_style_landmarks.extend([(0, style_h), (style_w, style_h)])
            mls_target_landmarks.extend([(0, tgt_h), (tgt_w, tgt_h)])

        # MLS deformation of style position guide toward target landmarks
        pos_tensor = ops._to_tensor(self.style_pos_guide).to(self.stylize_device)
        warped = ops.warp_mls_similarity(
            pos_tensor,
            torch.tensor(
                mls_style_landmarks, device=self.stylize_device, dtype=torch.float32
            ),
            torch.tensor(
                mls_target_landmarks, device=self.stylize_device, dtype=torch.float32
            ),
            grid_size=10,
        )
        target_pos_guide = ops._to_numpy_image(warped)

        # Resize using PyTorch
        target_pos_guide = _resize_torch(
            target_pos_guide, (target_img.shape[1], target_img.shape[0])
        )

        tgt_app = ops._ensure_grayscale(target_app_guide)
        stylization_rect = ops.get_head_area_rect(tgt_landmarks, target_img.shape[:2])

        # Primary stylization
        if patch_size > 0:
            stylized = ops.style_blit_voting(
                self.style_pos_guide,
                target_pos_guide,
                self.style_app_guide,
                tgt_app,
                self.look_up_cube,
                self.style_image,
                stylization_rect=stylization_rect,
                patch_size=patch_size,
                lambda_pos=10,  # Same as C++
                lambda_app=2,  # Same as C++
                device=self.stylize_device,
                use_vectorized=use_vectorized,
                dfs_mode=dfs_mode,
            )
        else:
            stylized = ops.style_blit(
                self.style_pos_guide,
                target_pos_guide,
                None,  # No appearance guide for direct mapping
                None,  # No appearance guide for direct mapping
                None,  # No LUT for direct mapping
                self.style_image,
                stylization_rect=stylization_rect,
            )

        # Background stylization (appearance disabled)
        if stylize_bg:
            stylized_bg = ops.style_blit_voting(
                self.style_pos_guide,
                target_pos_guide,
                None,
                None,
                None,
                self.style_image,
                stylization_rect=(0, 0, target_img.shape[1], target_img.shape[0]),
                patch_size=max(1, patch_size),
                lambda_pos=10,
                lambda_app=0,
                threshold=10,
                device=self.stylize_device,
                use_vectorized=use_vectorized,
                dfs_mode=dfs_mode,
            )
            result = ops.alpha_blend(
                stylized,
                stylized_bg,
                ops.get_skin_mask(target_img, tgt_landmarks),
                sigma=25.0,
            )
        else:
            result = ops.alpha_blend(
                stylized,
                target_img,
                ops.get_skin_mask(target_img, tgt_landmarks),
                sigma=25.0,
            )

        return result, target_pos_guide

    def stylize_image_file_with_guide_and_landmarks(
        self,
        input_path: PathLike,
        target_app_guide_path: PathLike,
        target_landmarks: Sequence[Tuple[int, int]],
        output_dir: PathLike | None = None,
        *,
        stylize_bg: bool = False,
        patch_size: int = 3,
        use_vectorized: bool = False,
    ) -> Path:
        """Stylize image from file using PIL/PyTorch for I/O."""
        inp = Path(input_path)
        out_dir = Path(output_dir) if output_dir is not None else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        image = _read_image_pil(inp)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {input_path}")

        # Read grayscale guide
        if Image is not None:
            guide_img = Image.open(_to_path(target_app_guide_path)).convert("L")
            guide = np.array(guide_img)
        else:
            raise RuntimeError("PIL is required to read images.")

        if guide is None:
            raise FileNotFoundError(
                f"Failed to read guide image: {target_app_guide_path}"
            )

        result, target_pos_guide = self.stylize_image_with_guide_and_landmarks(
            image,
            guide,
            target_landmarks,
            stylize_bg=stylize_bg,
            patch_size=patch_size,
            use_vectorized=use_vectorized,
        )
        output_path = out_dir / f"{inp.stem}_stylized.png"
        _write_image_pil(output_path, result)

        # Save target position guide
        target_pos_path = out_dir / f"{inp.stem}_target_pos.png"
        _write_image_pil(target_pos_path, target_pos_guide)

        return output_path

    # Compatibility stubs ------------------------------------------------
    def stylize_image(
        self, image_bgr: np.ndarray, stylize_bg: bool = False, patch_size: int = 3
    ) -> np.ndarray:
        raise NotImplementedError(
            "Provide guide and landmarks via stylize_image_with_guide_and_landmarks."
        )

    def stylize_image_with_guide(
        self,
        image_bgr: np.ndarray,
        target_app_guide: np.ndarray,
        stylize_bg: bool = False,
        patch_size: int = 3,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Provide landmarks via stylize_image_with_guide_and_landmarks."
        )

    def stylize_video_file(
        self,
        input_path: PathLike,
        output_dir: PathLike,
        stylize_bg: bool = False,
        patch_size: int = 3,
    ) -> None:
        raise NotImplementedError(
            "Video stylization requires per-frame landmarks and is not implemented."
        )

    def add_new_style(
        self,
        input_path: PathLike,
        output_dir: PathLike,
        *,
        predictor_path: PathLike | None = None,
        landmark_model: str = "dlib",
        lut_path: PathLike | None = None,
        draw_grid: bool = False,
        stretch_hist: bool = True,
        lambda_pos: int = 1,
        lambda_app: int = 1,
    ) -> dict:
        return compute_style_assets(
            input_path=input_path,
            output_dir=output_dir,
            predictor_path=predictor_path,
            landmark_model=landmark_model,
            lut_path=lut_path,
            draw_grid=draw_grid,
            stretch_hist=stretch_hist,
            lambda_pos=lambda_pos,
            lambda_app=lambda_app,
            reuse_precomputed=False,
            device=self.precompute_device,
        )


# ----------------------------------------------------------------------
# Module-level helpers mirroring the original API
# ----------------------------------------------------------------------


def get_gradient(width: int, height: int, draw_grid: bool = False) -> np.ndarray:
    return ops.gradient_guide(width, height, draw_grid=draw_grid)


def gray_hist_matching(input_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    return ops.gray_hist_matching(input_gray, ref_gray)


def get_skin_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Tuple[int, int]]
) -> np.ndarray:
    return ops.get_skin_mask(image_bgr, landmarks)


def get_app_guide(image_bgr: np.ndarray, stretch_hist: bool = True) -> np.ndarray:
    return ops.get_app_guide(image_bgr, stretch_hist=stretch_hist)


# ----------------------------------------------------------------------
# Style asset computation (dlib)
# ----------------------------------------------------------------------


def _style_asset_paths(
    out_dir: Path, stem: str, lut_path: Path | None
) -> dict[str, Path]:
    default_lut_path = out_dir / f"{stem}_lut.bytes"
    lut_candidates: list[Path] = []
    if lut_path is not None:
        lut_candidates.append(lut_path)
    lut_candidates.append(default_lut_path)

    resolved_lut_path = lut_candidates[0]
    for candidate in lut_candidates:
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                resolved_lut_path = candidate
                break
        except OSError:
            continue

    return {
        "landmarks": out_dir / f"{stem}_landmarks.txt",
        "pos": out_dir / f"{stem}_style_pos.png",
        "app": out_dir / f"{stem}_style_app.png",
        "lut": resolved_lut_path,
    }


def _load_precomputed(paths: dict[str, Path]) -> dict[str, str] | None:
    required = [paths["landmarks"], paths["pos"], paths["app"], paths["lut"]]
    try:
        if all(p.exists() and p.stat().st_size > 0 for p in required):
            return {
                "landmarks_path": _to_path(paths["landmarks"]),
                "style_pos_guide_path": _to_path(paths["pos"]),
                "style_app_guide_path": _to_path(paths["app"]),
                "lut_path": _to_path(paths["lut"]),
            }
    except OSError:
        return None
    return None


def compute_style_assets(
    input_path: PathLike,
    output_dir: PathLike,
    predictor_path: PathLike | None = None,
    *,
    landmark_model: str = "dlib",
    lut_path: PathLike | None = None,
    draw_grid: bool = False,
    stretch_hist: bool = True,
    lambda_pos: int = 10,
    lambda_app: int = 2,
    reuse_precomputed: bool = True,
    device: str | torch.device | None = None,
) -> dict:
    """Compute style assets (guides, landmarks, LUT) using pure Python/PyTorch."""
    style_path = Path(input_path)
    if not style_path.exists():
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    # Load image once (BGR numpy array)
    img = _read_image_pil(style_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = style_path.stem
    paths = _style_asset_paths(out_dir, stem, Path(lut_path) if lut_path else None)

    if reuse_precomputed:
        cached = _load_precomputed(paths)
        if cached is not None:
            cached["style_path"] = _to_path(style_path)
            return cached

    if landmark_model == "dlib":
        if predictor_path is None:
            raise ValueError(
                "predictor_path (dlib shape predictor .dat) is required for dlib landmark detection"
            )
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(_to_path(predictor_path))
        dets = detector(img, 1)
        if len(dets) == 0:
            raise RuntimeError("No face detected in style image")
        shape = sp(img, dets[0])
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    elif landmark_model == "fan":
        # Determine device for FA
        fa_device = "cpu"
        if device is not None:
            d = torch.device(device)
            if d.type == "cuda":
                fa_device = "cuda"
            elif d.type == "mps":
                fa_device = "mps"

        # Get cached FAN model
        fa = _get_fan_model(fa_device)

        # Convert BGR -> RGB for FAN
        img_np = img[..., ::-1].copy()
        preds = fa.get_landmarks(img_np)

        if preds is None or len(preds) == 0:
            raise RuntimeError("No face detected in style image (FAN)")

        # Take the first face
        shape = preds[0]
        landmarks = [(int(p[0]), int(p[1])) for p in shape]

    else:
        raise ValueError(f"Unknown landmark_model: {landmark_model}")

    # Persist assets
    with paths["landmarks"].open("w") as f:
        f.write("68\n")
        for x, y in landmarks:
            f.write(f"{x} {y}\n")

    pos = ops.gradient_guide(img.shape[1], img.shape[0], draw_grid=draw_grid)
    _write_image_pil(paths["pos"], pos)

    app = ops.get_app_guide(img, stretch_hist=stretch_hist)
    # Write grayscale image
    if Image is not None:
        Image.fromarray(app).save(_to_path(paths["app"]))
    else:
        raise RuntimeError("PIL is required to write images.")

    lut = ops.compute_look_up_cube_optimized(
        pos, app, lambda_pos=lambda_pos, lambda_app=lambda_app, device=device
    )
    ops.save_look_up_cube(lut, paths["lut"])

    return {
        "style_path": _to_path(style_path),
        "landmarks_path": _to_path(paths["landmarks"]),
        "style_pos_guide_path": _to_path(paths["pos"]),
        "style_app_guide_path": _to_path(paths["app"]),
        "lut_path": _to_path(paths["lut"]),
    }
