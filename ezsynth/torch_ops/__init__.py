# ezsynth/torch_ops/__init__.py
"""
PyTorch operations for EBSynth image synthesis.

This package contains vectorized PyTorch implementations of the core
EBSynth algorithms that work across CUDA, MPS (Metal), and CPU backends.
"""

from ..utils.timer import SynthesisTimer
from .mask_ops import dilate_mask, evaluate_mask
from .omega_ops import compute_omega_scores, populate_omega_map, update_omega_map
from .patch_ops import (
    compute_patch_ncc_vectorized,
    compute_patch_ssd_vectorized,
    extract_patches,
)
from .patchmatch_ops import propagation_step, random_search_step, try_patch_batch
from .voting_ops import vote_plain, vote_weighted

__all__ = [
    # Patch operations
    "extract_patches",
    "compute_patch_ssd_vectorized",
    "compute_patch_ncc_vectorized",
    # Omega operations
    "populate_omega_map",
    "compute_omega_scores",
    "update_omega_map",
    # Mask operations
    "evaluate_mask",
    "dilate_mask",
    # Voting operations
    "vote_plain",
    "vote_weighted",
    # PatchMatch operations
    "try_patch_batch",
    "propagation_step",
    "random_search_step",
    # Timer
    "SynthesisTimer",
]
