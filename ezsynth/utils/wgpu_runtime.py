"""Single owner of the ``ezsynth_wgpu`` Rust module and its process-wide device.

Every WebGPU engine (``kernels.synth.wgpu``, ``kernels.imageops.wgpu``,
``kernels.fastblend.wgpu``) imports the module through here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The 'wgpu' backend needs the ezsynth-wgpu package. Install the wheel for your OS "
    "from https://github.com/FuouM/ReEzSynth/releases with `pip install <wheel>`, or "
    "build it from the repository with `pip install ./rust/ezsynth-wgpu` / "
    "`uv sync --extra wgpu` (requires a Rust toolchain)."
)

_SYNTHESIZER = None


def import_wgpu():
    """Import ``ezsynth_wgpu`` or raise ImportError with install instructions."""
    try:
        import ezsynth_wgpu
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return ezsynth_wgpu


def shared_synthesizer():
    """Process-wide synthesizer (one WebGPU device, cached pipelines)."""
    global _SYNTHESIZER
    if _SYNTHESIZER is None:
        _SYNTHESIZER = import_wgpu().Synthesizer()
        logger.info(f"wgpu backend on {_SYNTHESIZER.adapter}")
    return _SYNTHESIZER
