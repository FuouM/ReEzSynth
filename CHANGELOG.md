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
- Synthesis options `cost_function="sad"`, `search_mode="jfa"|"multicand"` (with `multicand_k` 2–8) and `precision="float16"|"bfloat16"` for the torch and Taichi backends (`run_img_synth.py` flags `--cost-function`, `--search-mode`, `--multicand-k`, `--precision`). The native `cuda` backend rejects them with a clear error. Votes stay float32.
- ComfyUI: FastBlend Interpolate and FaceBlit nodes with example workflows; ComfyUI progress bar, Cancel support and model offloading before synthesis; nodes that need a missing optional package stop with the `pip install` command.
- `wgpu` backend for EbSynth and FastBlend (`rust/`, Rust + WebGPU via wgpu 30): runs on Vulkan, DX12 and Metal with any GPU vendor and no CUDA toolkit. Implements every option of the Taichi backend (ssd/ncc/sad, bilateral, `n_size_step`, modulation, patchmatch/jfa/multicand, weighted/plain votes) with bit-identical votes and error maps on the deterministic path; about 10x faster than Taichi and faster than the native CUDA extension. Python module `ezsynth_wgpu` (abi3 wheels on GitHub Releases, or `uv sync --extra wgpu` from source); config/CLI/node backend option `wgpu`; FastBlend `auto` prefers it after the native extension (ComfyUI node: first when installed).
- With the `wgpu` backend the video pipeline's image ops run on WebGPU too: backward flow warping (OpenCV 5 `remap` results), forward splatting and pull-push hole filling (Taichi kernel results), histogram blending (OpenCV path; Lab conversion within two 8-bit levels) and flow occlusion masks (identical to OpenCV). Forward warping no longer needs Taichi there. `ezsynth_wgpu` exposes them as `warp`, `forward_splat`, `pull_push`, `hist_blend` and `occlusion_masks`; the `warp_video` tools take `--image-ops taichi|wgpu`.
- Browser front end (`web/`, `rust/ebsynth-web`): Solid + Vite app on the SoulWindow design system with three WebGPU modes: single-image EbSynth, FastBlend keyframe interpolation and FastBlend deflicker of a stylized video. Video is decoded and encoded in the browser (WebCodecs via mediabunny); results save as MP4/WebM, a PNG zip or single frames. `ebsynth-web` exposes the FastBlend pyramid matcher (`WebSynth.fastBlend`) on the same device; the runners are ported to TypeScript with unit tests. Built with `npm run wasm && npm run build` in `web/`; CI builds and checks it. Keyframe mode can also run EbSynth per frame with the video pipeline's defaults (NCC cost; edge, colour, positional and temporal guides; forward/backward passes combined like ezsynth's Blender: lower-error selection, mask propagation along flow, Lab histogram blend and Poisson fusion; optional FastBlend deflicker), with NeuFlow optical flow through ONNX Runtime Web (`web/export_neuflow.py` exports the model); drag and drop, a before/after split view and remembered settings.
- Local Rust toolchain scripts (`setup_env.ps1`, `env.ps1`); CI checks rustfmt, clippy (native and wasm32) and WGSL validation, builds `ezsynth-wgpu` wheels, and attaches them to the GitHub Release of each pushed `v*` tag.

### Changed
- Works with NumPy 2, OpenCV 5, current PyTorch (checkpoints load with `weights_only=True`).
- `import ezsynth.api` no longer imports phycv, torchvision, scipy, taichi or pyamg.
- RAFT autocast follows the input device and its `mixed_precision` flag.
- Example media tracked with Git LFS.
- Repository formatted with ruff 0.17; wider lint rule set.
- Native extensions: single lazy loader (`ezsynth.native`); kernels launch on PyTorch's current stream with device guards and launch checks; no per-level device sync; `.cu` files are torch-free (nvcc C++17) while host glue uses PyTorch's C++ standard, so CUDA 12.4 builds against torch ≥ 2.12; portable host flags (`EZSYNTH_NATIVE_ARCH=1` opts in to native tuning); `ninja` is a dependency.
- FastBlend `is_available()` returns False (cached) instead of raising when its CUDA build fails.
- ComfyUI nodes and example workflows default to the Taichi backend (about 10x faster than pure PyTorch at full resolution); `taichi` is in `requirements.txt`. The video workflow uses ComfyUI's core video load/save nodes.
- Initial NNF is seeded (2024), so repeated runs give identical output.
- Torch patch planner always keeps `EZSYNTH_GPU_MEMORY_RESERVE_MB` free (also inside ComfyUI, where no allocator cap is set) and materializes patch grids only up to 2 GiB. The bounded strategy computes dense SSD/SAD/NCC costs straight from pixel gathers (same values, ~20% faster than before).
- `phycv` is optional for ComfyUI installs (only the PAGE/PST edge methods use it).
- `Warp` / `PositionalGuide` take `engine="opencv"|"taichi"|"wgpu"` instead of `use_taichi`; `hist_blender` takes `engine` instead of `use_taichi`. Forward splatting on the OpenCV engine raises instead of silently falling back to backward warping, and a missing Taichi raises instead of switching to OpenCV.

### Fixed
- `run_img_synth.py` passed `GuideObject`s where `ImageSynth.run` expects `(source, target, weight)` tuples.
- RAFT on CUDA machines with `device="cpu"` (DataParallel wrapper).
- Stale FastBlend Taichi parity "test" that never asserted; stale warp test.
- NaN optical flow from RAFT on inputs smaller than ~128 px (division by zero at 1-pixel correlation levels).
- `ezsynth/ebsynth_extension/dispatch.cpp` had been overwritten by a copy of `dispatch.h`, so the native extension could not link.
- FastBlend pyramid matchers crashed on construction (property called as a function).
- FaceBlit landmark detection allocated ~9 GiB of cuDNN workspace under PyTorch's `cudaMallocAsync` allocator (ComfyUI's default); cuDNN is disabled around FAN inference (peak now ~0.8 GiB).
- Taichi bilateral voting produced a black first estimate at every pyramid level (the initial vote's output aliased its centre-colour input).
- FastBlend temporal tracking (`tracking_window_size > 0`) crashed on the backward shift (batch-size mismatch).
- Taichi backward warping of multi-channel images failed to compile (a conditional expression evaluated both array ranks); the Taichi push pass read past the coarse level on odd image sizes.
