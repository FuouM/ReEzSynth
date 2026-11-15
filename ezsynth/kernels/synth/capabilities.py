"""Synthesis options each backend implements.

Pure data (no torch or backend imports): ``kernels.synth.create`` and the ComfyUI
nodes (``reezsynth_nodes/options.py``) check option combinations against it.
"""

from __future__ import annotations

COST_FUNCTIONS = ("ssd", "ncc", "sad")
SEARCH_MODES = ("patchmatch", "jfa", "multicand")
PRECISIONS = ("float32", "float16", "bfloat16")

BACKEND_COST_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "wgpu": COST_FUNCTIONS,
    "torch": COST_FUNCTIONS,
    "taichi": COST_FUNCTIONS,
    "cuda": ("ssd", "ncc"),
}
BACKEND_SEARCH_MODES: dict[str, tuple[str, ...]] = {
    "wgpu": SEARCH_MODES,
    "torch": SEARCH_MODES,
    "taichi": SEARCH_MODES,
    "cuda": ("patchmatch",),
}
# Storage dtype of the patch grids; only the torch backend has reduced precision.
BACKEND_PRECISIONS: dict[str, tuple[str, ...]] = {
    "wgpu": ("float32",),
    "torch": PRECISIONS,
    "taichi": ("float32",),
    "cuda": ("float32",),
}


def _backends_with(table: dict[str, tuple[str, ...]], value: str) -> str:
    return ", ".join(b for b, values in table.items() if value in values)


def unsupported_option(
    backend: str, cost_function: str, search_mode: str, precision: str = "float32"
) -> str | None:
    """Error message when ``backend`` lacks one of the options, else ``None``."""
    for option, value, table in (
        ("cost_function", cost_function, BACKEND_COST_FUNCTIONS),
        ("search_mode", search_mode, BACKEND_SEARCH_MODES),
        ("precision", precision, BACKEND_PRECISIONS),
    ):
        supported = table[backend]
        if value not in supported:
            return (
                f"{option}={value!r} is not supported by the {backend!r} backend "
                f"(supported: {', '.join(supported)}); use one of: "
                f"{_backends_with(table, value)}."
            )
    return None
