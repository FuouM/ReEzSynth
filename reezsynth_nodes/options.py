"""Choice lists, backend capability checks and input parsing for the ReEzSynth nodes.

Pure Python: importable without ComfyUI, torch or any optional dependency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ezsynth.kernels.synth import capabilities

SYNTH_BACKENDS = tuple(capabilities.BACKEND_COST_FUNCTIONS)
COST_FUNCTIONS = capabilities.COST_FUNCTIONS
SEARCH_MODES = capabilities.SEARCH_MODES
PRECISIONS = capabilities.PRECISIONS
VOTE_MODES = ("weighted", "plain")
MULTICAND_K_RANGE = (2, 8)

FLOW_ENGINES = ("RAFT", "NeuFlow", "OpenCV", "TorchVision")
FLOW_MODELS = (
    "sintel",
    "kitti",
    "small",
    "neuflow_mixed",
    "neuflow_sintel",
    "neuflow_things",
)
EDGE_METHODS = ("Classic", "PAGE", "PST")
OPENCV_FLOW_METHODS = ("DIS", "FARNEBACK")
TORCHVISION_FLOW_MODELS = ("raft_large", "raft_small")
POISSON_SOLVERS = ("disabled", "lsqr", "lsmr", "cg", "amg", "seamless", "taichi-cg")
# Poisson solvers built on scipy.sparse (lsqr runs on NumPy).
SCIPY_POISSON_SOLVERS = ("lsmr", "cg", "amg")

FACEBLIT_BACKENDS = ("torch", "taichi")
FACEBLIT_LANDMARK_DETECTORS = ("fan_blazeface", "fan_sfd")
FACEBLIT_DFS_MODES = ("numba", "python")

FASTBLEND_BACKENDS = ("auto", "wgpu", "taichi", "cupy", "cuda")
FASTBLEND_ACCURACY = {"fast": 1, "balanced": 2, "accurate": 3}

# The Rust WebGPU module: prebuilt wheels are GitHub Release assets; this source
# directory builds it with a Rust toolchain.
WGPU_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "rust" / "ezsynth-wgpu"
WGPU_RELEASES_URL = "https://github.com/FuouM/ReEzSynth/releases"

# Optional dependency -> pip requirement and what needs it.
OPTIONAL_DEPENDENCIES = {
    "taichi": ("taichi", "the 'taichi' synthesis / FaceBlit / FastBlend backend"),
    "ezsynth_wgpu": (
        f'"{WGPU_PACKAGE_DIR}"',
        f"the 'wgpu' synthesis / FastBlend backend (prebuilt ezsynth_wgpu wheels: "
        f"{WGPU_RELEASES_URL}; building from source needs a Rust toolchain)",
    ),
    "cupy": ("cupy-cuda12x", "the 'cupy' FastBlend backend"),
    "ninja": ("ninja", "JIT-building the native 'cuda' synthesis / FastBlend backend"),
    "scipy": ("scipy", "the lsmr / cg / amg Poisson solvers"),
    "phycv": ("phycv", "the PAGE / PST edge methods"),
    "pyamg": ("pyamg", "the 'amg' Poisson solver"),
    "face_alignment": ("face-alignment", "automatic FaceBlit landmark detection"),
    "numba": ("numba", "the FaceBlit CPU DFS path with dfs_mode='numba'"),
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_dependency_message(module: str) -> str:
    requirement, purpose = OPTIONAL_DEPENDENCIES[module]
    return (
        f"ReEzSynth: {purpose} needs the optional package '{requirement}', which is "
        f"not installed. Install it into ComfyUI's Python, e.g. "
        f"`python -m pip install {requirement}` (portable: "
        f"`python_embeded\\python.exe -m pip install {requirement}`)."
    )


def require_module(module: str) -> None:
    """Raise ``ImportError`` with an install hint when ``module`` is missing."""
    if not has_module(module):
        raise ImportError(missing_dependency_message(module))


def validate_synthesis_options(
    backend: str,
    cost_function: str,
    search_mode: str,
    precision: str = "float32",
    multicand_k: int = 3,
) -> str | None:
    """Return an error message for an unsupported combination, else ``None``."""
    if backend not in SYNTH_BACKENDS:
        return (
            f"Unknown backend {backend!r}; expected one of {', '.join(SYNTH_BACKENDS)}."
        )
    if precision not in PRECISIONS:
        return (
            f"Unknown precision {precision!r}; expected one of {', '.join(PRECISIONS)}."
        )
    error = capabilities.unsupported_option(
        backend, cost_function, search_mode, precision
    )
    if error is not None:
        return error
    low, high = MULTICAND_K_RANGE
    if not low <= int(multicand_k) <= high:
        return f"multicand_k must be in [{low}, {high}], got {multicand_k}."
    return None


def check_synthesis_options(
    backend: str,
    cost_function: str,
    search_mode: str,
    precision: str = "float32",
    multicand_k: int = 3,
) -> None:
    error = validate_synthesis_options(
        backend, cost_function, search_mode, precision, multicand_k
    )
    if error is not None:
        raise ValueError(error)
    if backend == "taichi":
        require_module("taichi")
    if backend == "wgpu":
        require_module("ezsynth_wgpu")
    if backend == "cuda":
        require_module("ninja")


def check_patch_size(patch_size: int) -> None:
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}.")


def parse_indices(text: str, name: str = "indices") -> list[int]:
    """Parse ``"0, 24, 48"`` (commas, spaces or newlines) into frame indices."""
    parts = text.replace(",", " ").split()
    try:
        indices = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            f"{name} must be frame numbers separated by commas, e.g. 0, 24, 48 "
            f"(got {text!r})."
        ) from exc
    if not indices:
        raise ValueError(f"{name} needs at least one frame number.")
    if any(index < 0 for index in indices):
        raise ValueError(f"{name} cannot contain negative frame numbers.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not repeat a frame number (got {indices}).")
    return indices


def check_indices_in_range(indices: list[int], frame_count: int, name: str) -> None:
    out_of_range = [i for i in indices if i >= frame_count]
    if out_of_range:
        raise ValueError(
            f"{name} {out_of_range} out of range: the content has {frame_count} "
            f"frames (valid 0-{frame_count - 1})."
        )


def parse_landmarks(text: str, name: str) -> list[tuple[int, int]] | None:
    """Parse 68 ``x y`` landmark pairs; ``None`` for empty text.

    Accepts the FaceBlit ``.txt`` layout (optional leading ``68`` count line, one
    ``x y`` pair per line) and flat comma/space separated lists ``x1,y1,x2,y2,...``.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0] == "68":
        lines = lines[1:]
    tokens = " ".join(lines).replace(",", " ").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            f"{name}: expected numbers as 'x y' pairs, got non-numeric text."
        ) from exc
    if len(values) != 136:
        raise ValueError(
            f"{name}: expected 68 'x y' landmark pairs (136 numbers), got {len(values)} numbers."
        )
    return [(round(values[i]), round(values[i + 1])) for i in range(0, 136, 2)]
