# Changelog

## Unreleased — modernization

### Breaking
- Environment managed by `uv` (`pyproject.toml` + `uv.lock`); `setup.py` removed. Supported Python: 3.10–3.12.
- ComfyUI node package renamed `comfy/` → `reezsynth_nodes/` (it shadowed ComfyUI's own `comfy` package). Node class names and workflows are unchanged.
- Command-line scripts moved into `ezsynth.cli` and exposed as console scripts:
  `run_video.py` → `reezsynth-video`, `prepare_video.py` → `reezsynth-prepare`,
  `fastblend_standalone.py` → `fastblend`, `fastblend_standalone_interpolate.py` → `fastblend-interpolate`.
- FaceBlit is a package: run `python -m FaceBlit.src.faceblit_run` from the repository root (no more `sys.path` edits).
- Removed unused vendored RAFT helpers (`ezsynth/raft/OpticalFlow.py`, `ezsynth/raft/warp.py`, `forward_interpolate`).
- Native extension sources moved `ebsynth_extension/` → `ezsynth/ebsynth_extension/`; `ebsynth_torch_loader.py` removed (use `ezsynth.native.ensure_extension_loaded()`).
- The project installs as a package (`uv sync` installs it editable); the root `__init__.py` remains only as the ComfyUI entry point.
- The prototype `ebsynth_torch` wheel in Releases no longer matches the supported PyTorch; wheels without `compiled_with_cuda()` are treated as CPU-only.

### Added
- GPU memory budget: outside ComfyUI, the PyTorch CUDA allocator is capped from the memory free at startup, keeping 1.5 GiB for other applications (`EZSYNTH_GPU_MEMORY_*` overrides).
- Torch backend picks a memory-bounded patch strategy automatically when full-image patch tensors would not fit the budget (`EZSYNTH_TORCH_PATCH_STRATEGY=auto|materialized|bounded`, `EZSYNTH_TORCH_PATCH_BUDGET_MB`). Stylit/Face Style examples now run in ~2 GiB instead of OOMing above 10 GiB.
- Library messages go through `logging` (CLIs configure it; `EZSYNTH_LOG_LEVEL` overrides the level).
- `ezsynth.utils.taichi_runtime`: one Taichi runtime shared by ezsynth, FastBlend and FaceBlit.
- CI (ruff + pytest on Linux), pre-commit hooks, `NOTICE` for bundled third-party code, ComfyUI and development docs.

### Changed
- Works with NumPy 2, OpenCV 5, current PyTorch (checkpoints load with `weights_only=True`).
- `import ezsynth.api` no longer imports phycv, torchvision, scipy, taichi or pyamg.
- RAFT autocast follows the input device and its `mixed_precision` flag.
- Example media tracked with Git LFS.
- Repository formatted with ruff 0.17; wider lint rule set.
- Native extensions: single lazy loader (`ezsynth.native`); kernels launch on PyTorch's current stream with device guards and launch checks; no per-level device sync; `.cu` files are torch-free (nvcc C++17) while host glue uses PyTorch's C++ standard, so CUDA 12.4 builds against torch ≥ 2.12; portable host flags (`EZSYNTH_NATIVE_ARCH=1` opts in to native tuning); `ninja` is a dependency.
- FastBlend `is_available()` returns False (cached) instead of raising when its CUDA build fails.

### Fixed
- `run_img_synth.py` passed `GuideObject`s where `ImageSynth.run` expects `(source, target, weight)` tuples.
- RAFT on CUDA machines with `device="cpu"` (DataParallel wrapper).
- Stale FastBlend Taichi parity "test" that never asserted; stale warp test.
- NaN optical flow from RAFT on inputs smaller than ~128 px (division by zero at 1-pixel correlation levels).
- `ezsynth/ebsynth_extension/dispatch.cpp` had been overwritten by a copy of `dispatch.h`, so the native extension could not link.
- FastBlend pyramid matchers crashed on construction (property called as a function).
