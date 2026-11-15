"""Compute kernels, grouped by operation family, one module per backend.

Each family package holds every implementation of its operations side by side, under
the same function names, so all versions of one operation sit in one directory:

- ``synth``: EbSynth level synthesis (``cuda``, ``torch``, ``taichi``, ``wgpu``);
  ``synth/STAGES.md`` maps each stage to its function in every backend.
- ``imageops``: flow warping, forward splatting, pull-push hole filling, histogram
  blending, occlusion masks, Taichi Poisson CG (``opencv``, ``taichi``, ``wgpu``).
- ``fastblend``: FastBlend pyramid patch matching (``cuda``, ``cupy``, ``taichi``,
  ``wgpu``).

The family names match the Rust crate ``rust/ebsynth-gpu`` (``ebsynth.rs``,
``imageops.rs``, ``fastblend.rs`` and their ``shaders/*.wgsl``).
"""
