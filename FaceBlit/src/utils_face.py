"""68-point facial landmark detection (dlib or face-alignment FAN).

Both detector libraries are optional and imported only when a detector is used.
"""

import inspect
import logging
import threading
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# Face detectors understood by face-alignment's FAN: ``blazeface`` (small, fast),
# ``sfd`` (S3FD, more robust, larger) and ``dlib`` (needs the dlib package).
FAN_FACE_DETECTORS = ("blazeface", "sfd", "dlib")
DEFAULT_FAN_FACE_DETECTOR = "blazeface"

FACE_ALIGNMENT_INSTALL_HINT = (
    "Landmark detection needs the 'face-alignment' package "
    "(pip install face-alignment). Its FAN and face detector weights are "
    "downloaded to the torch hub cache on first use."
)
DLIB_INSTALL_HINT = (
    "The dlib landmark backend needs the 'dlib' package (pip install dlib) and "
    "FaceBlit/models/shape_predictor_68_face_landmarks.dat."
)

_lock = threading.Lock()
_dlib_detector: object | None = None
_dlib_shape_predictors: dict[str, object] = {}
_fan_model_cache: dict[tuple[str, str], object] = {}


def detect_landmarks(
    image_bgr,
    backend: str,
    predictor_path: Path | None,
    device: str,
    face_detector: str = DEFAULT_FAN_FACE_DETECTOR,
) -> list[tuple[int, int]] | None:
    """Detect 68 landmarks on a BGR image; ``None`` when no face is found.

    ``backend`` is ``"dlib"`` (needs ``predictor_path``) or ``"fan"``
    (face-alignment with ``face_detector``).
    """
    if backend == "dlib":
        if predictor_path is None:
            raise ValueError("The dlib landmark backend needs a predictor_path.")
        return detect_landmarks_dlib(image_bgr, predictor_path)
    if backend == "fan":
        return detect_landmarks_fan(image_bgr, device, face_detector)
    raise ValueError(f"Unknown landmark backend {backend!r}; expected 'dlib' or 'fan'.")


def detect_landmarks_dlib(image_bgr, predictor_path):
    global _dlib_detector
    try:
        import dlib
    except ImportError as exc:
        raise ImportError(DLIB_INSTALL_HINT) from exc

    path_str = str(predictor_path)
    with _lock:
        sp = _dlib_shape_predictors.get(path_str)
        if sp is None:
            sp = dlib.shape_predictor(path_str)
            _dlib_shape_predictors[path_str] = sp
        if _dlib_detector is None:
            _dlib_detector = dlib.get_frontal_face_detector()
        detector = _dlib_detector
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)
    if len(faces) == 0:
        return None
    shape = sp(img_rgb, faces[0])
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_landmarks_fan(
    image_bgr, device: str, face_detector: str = DEFAULT_FAN_FACE_DETECTOR
):
    fa = get_fan_model(device, face_detector)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    import torch

    # cuDNN's conv algorithm choice for these small nets grabs ~9 GiB of workspace
    # when PyTorch uses the cudaMallocAsync allocator (ComfyUI's default); the native
    # convolution kernels need a few hundred MiB and are fast enough for a few faces.
    with torch.backends.cudnn.flags(enabled=False):
        preds = fa.get_landmarks(img_rgb)
    if preds is None or len(preds) == 0:
        return None
    shape = preds[0]
    return [(int(p[0]), int(p[1])) for p in shape]


def get_fan_model(device: str, face_detector: str = DEFAULT_FAN_FACE_DETECTOR):
    """Cached face-alignment 2D FAN model for ``(device, face_detector)``."""
    if face_detector not in FAN_FACE_DETECTORS:
        raise ValueError(
            f"Unknown FAN face detector {face_detector!r}; "
            f"expected one of {', '.join(FAN_FACE_DETECTORS)}."
        )
    key = (str(device), face_detector)
    with _lock:
        cached = _fan_model_cache.get(key)
        if cached is not None:
            return cached

        try:
            import face_alignment
        except ImportError as exc:
            raise ImportError(FACE_ALIGNMENT_INSTALL_HINT) from exc

        logger.info(f"[FAN] Loading model ({face_detector}) for device: {device}...")
        start_time = time.time()

        # LandmarksType renamed _2D -> TWO_D in face-alignment 1.4.
        landmarks_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
        if landmarks_type is None:
            landmarks_type = getattr(face_alignment.LandmarksType, "_2D", None)
        if landmarks_type is None:
            raise ImportError(
                "face_alignment.LandmarksType has neither TWO_D nor _2D; "
                "install face-alignment>=1.3."
            )

        kwargs = {
            "device": str(device),
            "flip_input": True,
            "face_detector": face_detector,
        }
        # face-alignment>=1.5 torch.compile()s FAN by default, which needs a working
        # C++ toolchain (with OpenMP) at runtime; eager mode is fast enough for stills.
        if "compile" in inspect.signature(face_alignment.FaceAlignment).parameters:
            kwargs["compile"] = False
        fa = face_alignment.FaceAlignment(landmarks_type, **kwargs)

        logger.info(
            f"[FAN] Model loaded for device {device} with {face_detector} "
            f"in {time.time() - start_time:.3f}s"
        )
        _fan_model_cache[key] = fa
        return fa
