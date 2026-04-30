import datetime
import time
from pathlib import Path

import cv2
import dlib
import face_alignment

FAN_MODEL = "dlib"

## Supported FAN models
# sfd
# dlib
# blazeface

_dlib_detector: object | None = None
_dlib_shape_predictors: dict[str, object] = {}


def _log_timestamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def detect_landmarks(
    image_bgr, backend: str, predictor_path: Path, device: str
) -> list[tuple[int, int]] | None:
    """Detect landmarks on a BGR image with the selected backend."""
    if backend == "dlib":
        return detect_landmarks_dlib(image_bgr, predictor_path)
    return detect_landmarks_fan(image_bgr, predictor_path, device)


def detect_landmarks_dlib(image_bgr, predictor_path, device: str | None = None):
    global _dlib_detector
    path_str = str(predictor_path)
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


def detect_landmarks_fan(image_bgr, predictor_path: Path, device: str | None = None):
    fa = get_fan_model(device, FAN_MODEL)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(img_rgb)
    if preds is None or len(preds) == 0:
        return None
    shape = preds[0]
    return [(int(p[0]), int(p[1])) for p in shape]


_fan_model_cache: dict[str, object] = {}


def get_fan_model(device: str, model: str = "dlib"):
    """Get cached FAN model or create new one for the specified device."""
    if device in _fan_model_cache:
        print(f"[{_log_timestamp()}] [FAN] Using cached model for device: {device}")
        return _fan_model_cache[device]

    print(f"[{_log_timestamp()}] [FAN] Loading model for device: {device}...")

    start_time = time.time()

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
        f"[{_log_timestamp()}] [FAN] Model loaded for device: {device} with {model} in {load_time:.3f}s"
    )
    _fan_model_cache[device] = fa
    return fa
