"""ComfyUI integration: device choice, VRAM hand-off, progress and interruption.

Imported only inside node execution (``comfy`` exists only in a ComfyUI process).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

_MIB = 1 << 20


def torch_device() -> torch.device:
    """The device ComfyUI runs models on (honours --cpu, --cuda-device, ...)."""
    import comfy.model_management as mm

    return mm.get_torch_device()


def prepare_vram(device: torch.device, required_bytes: int) -> None:
    """Ask ComfyUI to unload / offload its models until ``required_bytes`` are free."""
    if device.type != "cuda":
        return
    import comfy.model_management as mm

    mm.free_memory(int(required_bytes), device)
    mm.soft_empty_cache()


def release_vram() -> None:
    """Return cached blocks to the driver so ComfyUI models can use them again."""
    import comfy.model_management as mm

    gc.collect()
    mm.soft_empty_cache()


def throw_if_interrupted() -> None:
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def estimate_synthesis_bytes(height: int, width: int, guide_channels: int) -> int:
    """Rough peak VRAM for one EbSynth pass at full resolution.

    Covers style/guide pyramids, NNF + error buffers and the patch-cost scratch the
    torch backend allocates per level, with headroom for the allocator.
    """
    pixels = height * width
    per_pixel = 4 * (2 * (3 + guide_channels) + 16) + 7 * 7 * 4 * 4
    return 512 * _MIB + pixels * per_pixel


@contextmanager
def comfy_progress(
    node_id: str | None = None,
) -> Iterator[Callable[[str, int, int], None]]:
    """Install an ezsynth progress hook driving a ComfyUI ProgressBar.

    Every report also checks ComfyUI's interrupt flag, so "Cancel" stops synthesis
    at the next frame / pyramid-level boundary. Yields the reporter so callers with
    their own loops (FastBlend, FaceBlit) can call it directly.
    """
    import comfy.utils

    from ezsynth.utils.progress import progress_hook

    bar = comfy.utils.ProgressBar(1, node_id=node_id)
    state = {"stage": None}

    def report(stage: str, current: int, total: int) -> None:
        throw_if_interrupted()
        if stage != state["stage"]:
            state["stage"] = stage
            logger.debug("ReEzSynth stage: %s (%d steps)", stage, total)
        bar.update_absolute(min(current, total), max(total, 1))

    with progress_hook(report):
        yield report
