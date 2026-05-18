"""Lightweight checks for optional third-party backends (no heavy imports)."""

import importlib.util


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


cupy_available: bool = _has("cupy")
taichi_available: bool = _has("taichi")
