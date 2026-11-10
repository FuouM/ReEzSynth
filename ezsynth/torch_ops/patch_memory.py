"""Memory planning for the PyTorch synthesis backend.

The fast path materializes every patch grid with ``F.unfold`` (H*W*C*patch_size**2
values per grid, float32 or the configured low-precision dtype). Large guide stacks
make that exceed the process GPU budget,
so each pyramid level picks a strategy:

- ``materialized``: the unfold grids (fastest) when the estimate fits the budget;
- ``bounded``: :class:`~ezsynth.torch_ops.patch_ops.ImagePatches` views that gather
  patches for chunks of target pixels, with the chunk size derived from the budget.

Environment overrides (read per level):

- ``EZSYNTH_TORCH_PATCH_STRATEGY``: ``auto`` (default), ``materialized`` or ``bounded``.
- ``EZSYNTH_TORCH_PATCH_BUDGET_MB``: memory budget in MiB used instead of the measured
  free device memory (applies to every device type).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..consts import vote_chunk_budget_bytes
from ..utils.gpu_memory import current_budget_bytes, reserve_bytes

STRATEGY_ENV = "EZSYNTH_TORCH_PATCH_STRATEGY"
BUDGET_ENV = "EZSYNTH_TORCH_PATCH_BUDGET_MB"
STRATEGIES = ("auto", "materialized", "bounded")

_MIB = 1 << 20
# Kept free on top of the estimate for allocator fragmentation and small temporaries.
SAFETY_MARGIN_BYTES = 256 * _MIB
# Fraction of the measured budget the materialized estimate may use.
MATERIALIZED_FRACTION = 0.8
# Absolute ceiling for the materialized estimate on CUDA. Above it the bounded strategy
# is about as fast (kernels already saturate the GPU) and grabbing many GiB just
# because they are free right now starves the desktop and other GPU apps.
MAX_MATERIALIZED_BYTES = 2 << 30
# Fraction of the budget one bounded chunk may use, and its absolute ceiling
# (larger chunks stop paying off once kernels saturate the device).
CHUNK_FRACTION = 0.5
MAX_CHUNK_BYTES = 1 << 30
_FLOAT_BYTES = 4
_INDEX_BYTES = 8


@dataclass(frozen=True)
class LevelShape:
    """Sizes that determine patch memory of one pyramid level.

    ``patch_element_bytes`` is the storage size of one patch value (4 for float32,
    2 for float16/bfloat16); ``candidate_slots`` is the number of multicand
    alternative matches kept per target pixel (0 for other search modes).
    """

    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    style_channels: int
    guide_channels: int
    modulation_channels: int
    patch_size: int
    patch_element_bytes: int = _FLOAT_BYTES
    candidate_slots: int = 0


@dataclass(frozen=True)
class PatchPlan:
    """Chosen strategy; ``chunk_pixels`` is None for the materialized strategy."""

    strategy: str
    estimate_bytes: int
    budget_bytes: int | None
    chunk_pixels: int | None


def strategy_override() -> str:
    raw = os.environ.get(STRATEGY_ENV, "").strip().lower() or "auto"
    if raw not in STRATEGIES:
        raise ValueError(f"{STRATEGY_ENV} must be one of {STRATEGIES}, got {raw!r}")
    return raw


def budget_override_bytes() -> int | None:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if not raw:
        return None
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{BUDGET_ENV} must be positive, got {raw!r}")
    return int(value * _MIB)


def available_bytes(device: torch.device | str) -> int | None:
    """
    Memory a new allocation on ``device`` can still use, or None when unknown.

    CUDA: the smaller of physically free memory plus this process's cached-but-unused
    blocks minus the reserve kept for other applications
    (:func:`ezsynth.utils.gpu_memory.reserve_bytes`; applies even where no allocator cap
    is set, e.g. inside ComfyUI), and the per-process budget minus the memory already
    allocated. CPU/MPS share system memory and report None.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    free, _total = torch.cuda.mem_get_info(index)
    allocated = torch.cuda.memory_allocated(index)
    reserved = torch.cuda.memory_reserved(index)
    available = free + max(0, reserved - allocated) - reserve_bytes()
    cap = current_budget_bytes(torch.device("cuda", index))
    if cap is not None:
        available = min(available, cap - allocated)
    return max(0, available - SAFETY_MARGIN_BYTES)


def candidate_set_bytes(shape: LevelShape) -> int:
    """
    Bytes of the multicand alternative set: int32 coordinates + float32 errors per
    slot, plus the same again twice for offer/vote temporaries (gathered slots,
    duplicate masks, per-slot int64 coordinate copies).
    """
    nt = shape.target_hw[0] * shape.target_hw[1]
    return 3 * nt * shape.candidate_slots * (2 * 4 + _FLOAT_BYTES)


def estimate_materialized_bytes(shape: LevelShape, *, iterative: bool) -> int:
    """
    Peak bytes of the unfold strategy for one level.

    Persistent grids: source style + source guide, target guide + modulation, and the
    target style grid (iterative: re-extracted every search-vote iteration; otherwise
    only when the style is resampled to a different target size), plus the multicand
    alternative set. Transient peak is the larger of one unfold (padded copy +
    contiguous result, ~2 grids) and a full-frame random search (target gathers for
    every active pixel plus float32 matched-source copies and modulation products).
    """
    ps2 = shape.patch_size * shape.patch_size
    eb = shape.patch_element_bytes
    cs = shape.style_channels * ps2
    cg = shape.guide_channels * ps2
    cm = shape.modulation_channels * ps2
    ns = shape.source_hw[0] * shape.source_hw[1]
    nt = shape.target_hw[0] * shape.target_hw[1]
    persistent = (ns * (cs + cg) + nt * (cg + cm)) * eb
    if iterative or shape.source_hw != shape.target_hw:
        persistent += nt * cs * eb
    persistent += candidate_set_bytes(shape)
    extraction = 2 * max(ns * max(cs, cg), nt * max(cs, cg, cm)) * eb
    search = nt * (cs + cg + cm) * eb + nt * (cs + cg + 2 * cm) * _FLOAT_BYTES
    return persistent + max(extraction, search)


def bounded_bytes_per_pixel(shape: LevelShape) -> int:
    """Transient bytes per target pixel of one bounded chunk.

    Gathered source and target patches, the cost kernels' own copies of the matched
    source patches (twice for style under bilateral SSD), modulation products, and
    the int64 pixel-index tensors with their construction temporaries. Counted as
    float32 regardless of the storage precision (the cost kernels upcast).
    """
    ps2 = shape.patch_size * shape.patch_size
    ds = shape.style_channels * ps2
    dg = shape.guide_channels * ps2
    dm = shape.modulation_channels * ps2
    floats = 4 * ds + 3 * dg + 3 * dm
    return floats * _FLOAT_BYTES + 6 * ps2 * _INDEX_BYTES


def plan_level(
    shape: LevelShape, device: torch.device | str, *, iterative: bool
) -> PatchPlan:
    """Pick materialized grids when they fit the budget, else bounded chunks.

    The multicand alternative set is persistent in both strategies, so the bounded
    chunks only get the budget left after it; :class:`MemoryError` when nothing is
    left (lower ``multicand_k`` or raise the budget).
    """
    strategy = strategy_override()
    estimate = estimate_materialized_bytes(shape, iterative=iterative)
    budget = budget_override_bytes()
    if budget is None:
        budget = available_bytes(device)
    if strategy == "auto":
        fits = budget is None or estimate <= min(
            budget * MATERIALIZED_FRACTION, MAX_MATERIALIZED_BYTES
        )
        strategy = "materialized" if fits else "bounded"
    if strategy == "materialized":
        return PatchPlan("materialized", estimate, budget, None)
    if budget is None:
        chunk_bytes = vote_chunk_budget_bytes()
    else:
        candidates = candidate_set_bytes(shape)
        remaining = budget - candidates
        if remaining <= 0:
            raise MemoryError(
                f"multicand alternatives for a {shape.target_hw[1]}x"
                f"{shape.target_hw[0]} level need ~{candidates >> 20} MiB but the "
                f"memory budget is {budget >> 20} MiB; lower multicand_k or raise "
                "the budget"
            )
        chunk_bytes = min(MAX_CHUNK_BYTES, int(remaining * CHUNK_FRACTION))
    chunk_pixels = max(1, chunk_bytes // bounded_bytes_per_pixel(shape))
    return PatchPlan("bounded", estimate, budget, chunk_pixels)
