"""Compute kernels, grouped by operation family, one module per backend.

Each family package holds every implementation of its operations side by side, under
the same function names, so all versions of one operation sit in one directory:

- ``imageops``: flow warping, forward splatting, pull-push hole filling, histogram
  blending, occlusion masks, Taichi Poisson CG (``opencv``, ``taichi``, ``wgpu``).

The family names match the Rust crate ``rust/ebsynth-gpu`` (``imageops.rs`` and
``shaders/imageops_*.wgsl``).
"""
