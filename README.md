# ReEzSynth: A PyTorch-Powered Remake of Ebsynth

<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD045 -->

ReEzSynth is a complete, from-the-ground-up rewrite and enhancement of the original Ebsynth video stylization tool. It leverages the power and flexibility of PyTorch and a custom CUDA extension to deliver high-performance, high-quality video-to-video synthesis, while adding new features to dramatically improve temporal stability.

This project is designed for artists, researchers, and developers who need a robust, scriptable, and high-performance tool for temporal style transfer.

## Key Features

- **High-Performance CUDA Backend**: The core patch-matching and synthesis algorithm is implemented as a PyTorch C++/CUDA extension, ensuring maximum performance on modern GPUs.
- **WebGPU Backend (Rust)**: `wgpu` runs EbSynth and FastBlend on Vulkan, DX12 or Metal with any GPU vendor and no CUDA toolkit; the same kernels run in the browser (see [Web](#web)).
- **Full Parity with Original Ebsynth**:
  - **Multi-Guide Synthesis**: Use multiple guide layers (e.g., edges, colors, positional maps) with individual weights.
  - **Modulation Maps**: Spatially control the influence of guides for fine-grained artistic direction.
  - **Weighted & Plain Voting**: Choose between sharp, detail-preserving synthesis (`weighted`) or a softer, more painterly style (`plain`).
- **New Advanced Features for Temporal Stability**:
  - **Temporal NNF Propagation (Experimental)**: Intelligently re-uses information from the previous frame to initialize the current one, drastically reducing flicker and improving detail preservation.
  - **Sparse Feature Guiding (Experimental)**: Automatically detects and tracks key features in the video, creating an internal guide that "pins" the style to moving objects and prevents sliding textures.
- **State-of-the-Art Pipeline**:
  - **Bidirectional Synthesis & Blending**: Processes video in both forward and reverse passes and seamlessly merges them with a choice of high-quality Poisson solvers to eliminate jitter.
  - **High-Quality Optical Flow**: Integrates modern optical flow models (RAFT, NeuFlow v2) for accurate motion tracking.
- **Modern and User-Friendly**:
  - **YAML-based Projects**: Easily define and manage complex projects with simple configuration files for the command-line workflow.
  - **Full Python API**: A clean and simple Python API (`Ezsynth`, `ImageSynth`) allows for easy integration into larger graphics pipelines and creative coding projects.
  - **JIT CUDA Compilation**: Native PyTorch JIT loader automatically compiles and caches the CUDA extension on-demand, eliminating complex build dependencies.
  - **Caching System**: Automatically caches expensive pre-computation steps (like optical flow) to accelerate iterative workflows.

## Examples

| Face style | Stylit | Retarget |
|:-:|:-:|:-:|
| <img src="output_synth_api/facestyle_out.png" height="250"> | <img src="output_synth_api/stylit_out.png" height="250"> | <img src="output_synth_api/retarget_out.png" height="250"> |

`run_img_synth.py` Total time taken: 2.7780 s (RTX 3060, Ryzen 5 3600)

<!-- <Video here> -->

<https://github.com/user-attachments/assets/0bdad993-e5ef-40bb-8c9a-47186e307fd5>

> Forward, Blend, Original and Reverse (clock-wise)

<https://github.com/user-attachments/assets/770dd426-b708-4436-bc28-9302dad06ce5>

> Seamless, AMG, LSMR, LSQR, CG, Disabled Blend methods (clock-wise)

Note: Pipeline Time includes saving to disk. Benchmarks run on an RTX 3060 and Ryzen 5 3600.

| Blend method | Pipeline time (s) | Poisson Recon time (mm:ss) |
|-|-|-|
| none (f/r)  | ~73 | N/A |
| disabled    | 147 | N/A |
| seamless    | 164 | 00:17 |
| cg          | 192 | 00:52 |
| amg         | 235 | 01:31 |
| lsmr        | 299 | 02:36 |
| lsqr        | 351 | 03:33 |

- **Ebsynth Single Pass time:** ~46 s
- **Histogram Blending time:** ~22 s
- **Settings:** 100 frames, 960x544 resolution, 2 style keyframes (first and last).

Different Loss functions

<https://github.com/user-attachments/assets/c909ebc9-b466-4907-9236-bf9e7447ea7c>

> Sum of Squared Difference (Top), Normalized Cross-Correlation (Bottom)

## Installation

### Prerequisites

1. **Python 3.10** (supported range: 3.10–3.12; Taichi is the upper bound).
2. **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.
3. **GPU (optional)**: an NVIDIA GPU is used automatically when available. The PyTorch and Taichi backends need no CUDA Toolkit.
4. **For the native `ebsynth_torch` extension only** (`cuda`/`cpu` backends): a C++20 compiler (Visual Studio 2022 "Desktop development with C++" on Windows, GCC 10+ on Linux) and, for its CUDA kernels, the NVIDIA CUDA Toolkit 12.x. `ninja` is installed with the project.

### Setup

1. **Clone the Repository**:

    ```bash
    git clone --filter=blob:none https://github.com/FuouM/ReEzSynth.git
    cd ReEzSynth
    ```

    `--filter=blob:none` makes a partial clone: file contents are downloaded only for the checked-out commit, so the large example media stored in older history is skipped (current media comes from Git LFS). To skip the example media entirely, prefix the clone with `GIT_LFS_SKIP_SMUDGE=1`, then fetch what you need later, e.g. `git lfs pull --include="examples/**"`.

2. **Create the environment**:
    `uv sync` creates `.venv` with Python 3.10 and installs the locked dependencies, including PyTorch built for CUDA 12.6 (CPU/MPS build on macOS).

    ```bash
    uv sync                                   # core
    uv sync --extra taichi --extra pyamg      # + Taichi backend, AMG Poisson solver
    uv sync --all-extras                      # everything, incl. CuPy and FaceBlit (dlib needs CMake)
    ```

    Optional extras: `taichi`, `wgpu` (builds the Rust WebGPU module; needs the Rust toolchain from step 4b), `cupy`, `pyamg`, `neuflow` (Hugging Face weight download), `faceblit`.

3. **Run** the command-line tools through the environment, e.g. `uv run reezsynth-video --config configs/example_project.yml`, and tests with `uv run pytest`. Console scripts: `reezsynth-video`, `reezsynth-prepare`, `fastblend`, `fastblend-interpolate` (each also runs as `python -m ezsynth.cli.<name>`).

4. **Native extension (optional)**:
    The `ebsynth_torch` C++/CUDA extension (`ezsynth/ebsynth_extension/`) is JIT-compiled on first use of the `cuda`/`cpu` backends and cached by PyTorch. Host code is compiled with PyTorch's C++ standard (C++20 for torch ≥ 2.12); `.cu` files include no PyTorch headers and are compiled by nvcc as C++17, so CUDA 12.4+ toolkits work. Builds are portable by default; CUDA architectures come from `TORCH_CUDA_ARCH_LIST` or the local GPU.

    ```bash
    export EZSYNTH_NATIVE_ARCH=1         # opt in to -march=native / /arch:AVX2 host tuning
    export FORCE_EBSYNTH_JIT_LOADER=1    # ignore an installed ebsynth_torch wheel and JIT-build
    ```

4b. **WebGPU backend (optional, Rust)**:
    `rust/` holds the WebGPU implementation (`ebsynth-gpu` core, `ezsynth-wgpu` Python module, `ebsynth-web` browser module). Prebuilt `ezsynth_wgpu` wheels (abi3: one per OS, for Python ≥ 3.10 and any PyTorch) are attached to each [GitHub Release](https://github.com/FuouM/ReEzSynth/releases); install one with `pip install ezsynth_wgpu-<version>-cp310-abi3-<platform>.whl`. To build from source instead, install a local Rust toolchain into `.rust/` with `setup_env.ps1` (Windows; elsewhere use [rustup](https://rustup.rs)), load it with `. .\env.ps1`, then `uv sync --extra wgpu` (or `pip install ./rust/ezsynth-wgpu`). Select it with `backend: wgpu` / `--backend wgpu`; FastBlend `auto` prefers it after the native extension. `WGPU_BACKEND=vulkan|dx12|metal` and `WGPU_ADAPTER_NAME` pick the graphics API and GPU. Pushing a `v*` tag makes CI build the wheels and publish them to that tag's Release.

    Full-pyramid run time per example image (RTX 3060, `run_img_synth.py` full parameters):

    | Backend | retarget 462×320 | stylit 912×1200 | facestyle 1000×810 |
    |:-|-:|-:|-:|
    | `wgpu` | 0.13 s | 0.97 s | 0.61 s |
    | `cuda` (native extension) | 0.17 s | 1.36 s | 0.85 s |
    | `taichi` | 1.51 s | 8.84 s | 6.27 s |

    In ComfyUI the `wgpu` backend runs the bundled video workflow (100 frames, 960×544) in 2 min 43 s versus 12 min 39 s with `taichi`, and FastBlend (98 frames) in 125 s versus 298 s.

    With `backend: wgpu` the video pipeline's image operations run on WebGPU as well: flow warping, forward splatting with hole filling (so `use_forward_warping` needs no Taichi), histogram blending and flow occlusion masks. At 960×544 (RTX 3060) blending 40 bidirectional frames takes 0.8 s instead of 13.4 s, and occlusion masks for 40 flow pairs 0.3 s instead of 6.5 s. Warping and occlusion masks match OpenCV 5; histogram blending converts to Lab in float, so a few pixels differ from OpenCV's 8-bit Lab path (mean difference below one level).

5. **GPU memory budget**:
    Outside ComfyUI, each run caps PyTorch's CUDA allocator to a budget computed from the memory free *at startup* (90% of it, always leaving 1.5 GiB for the browser/desktop). Running out of budget raises a CUDA out-of-memory error in ReEzSynth instead of exhausting the whole GPU. Overrides: `EZSYNTH_GPU_MEMORY_LIMIT_MB` (absolute cap), `EZSYNTH_GPU_MEMORY_FRACTION`, `EZSYNTH_GPU_MEMORY_RESERVE_MB`, or `EZSYNTH_GPU_MEMORY_LIMIT=0` to disable.

### ComfyUI

1. Clone into `ComfyUI/custom_nodes/ReEzSynth`, preferably as a partial clone without media: `GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/FuouM/ReEzSynth.git` (on Windows `cmd`: `set GIT_LFS_SKIP_SMUDGE=1` first).
2. Install the requirements into ComfyUI's Python (torch/torchvision come from ComfyUI); they include `taichi`, the nodes' default synthesis backend. Portable build: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ReEzSynth\requirements.txt`. Taichi has no wheels for Python 3.13+; there, select the `torch` backend (pure PyTorch, about 10x slower at full resolution).
3. For video synthesis with the default RAFT flow engine, put the RAFT checkpoints in `models/raft/` (see `models/model_files.md`). The TorchVision flow engine downloads its weights automatically; OpenCV needs none.
4. Restart ComfyUI. Nodes appear under the `ReEzSynth` prefix: Guide, Guides Append, Image Synth Config, Image Synth, Video Style Keyframe(s), Video Style Keyframes Append, Video Synth Config, Video Synth, FastBlend Interpolate, FaceBlit.
5. Example graphs are in `workflows/` (`reezsynth_imgsynth_facestyle.json`, `reezsynth_imgsynth_stylelit.json`, `reezsynth_imgsynth_texbynum.json`, `reezsynth_videosynth.json`, `reezsynth_fastblend.json`, `reezsynth_faceblit.json`); drag one onto the canvas.
6. Optional, fastest: install the `wgpu` backend into ComfyUI's Python, either the wheel for your OS from the [Releases](https://github.com/FuouM/ReEzSynth/releases) (`python_embeded\python.exe -m pip install ezsynth_wgpu-<version>-cp310-abi3-win_amd64.whl`) or a source build (`python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ReEzSynth\rust\ezsynth-wgpu`, needs a Rust toolchain), then pick `wgpu` in the config nodes. FastBlend `auto` uses it when installed.

Optional packages, needed only by the options that use them (a node that needs a missing one stops with the `pip install` command to run):

| Package | Needed for |
|:-|:-|
| `cupy-cuda12x` | FastBlend `cupy` backend (FastBlend `auto` uses it only when Taichi is missing) |
| `phycv` | PAGE / PST edge methods |
| `pyamg` | `amg` Poisson solver |
| `huggingface-hub` | NeuFlow flow engine (checkpoints in `models/neuflow/`) |
| `face-alignment` | FaceBlit landmark detection when no landmarks are connected (weights download on first use) |
| `numba` | FaceBlit `numba` DFS mode with the non-vectorized torch path |

The `cuda` synthesis and FastBlend backends JIT-build the native extension, which needs a C++ compiler and the CUDA Toolkit matching ComfyUI's PyTorch.

Inside ComfyUI the allocator cap above is disabled (ComfyUI manages VRAM for its process). The nodes ask ComfyUI to offload its models before synthesis, report progress to the ComfyUI progress bar and stop at the next frame or pyramid level when you cancel. The torch backend still sizes its patch tensors so that `EZSYNTH_GPU_MEMORY_RESERVE_MB` (1.5 GiB by default) stays free for the desktop.

A prototype `ebsynth_torch-0.0.0-cp310-cp310-win_amd64.whl` exists in the Releases (SHA256 `180c6eeb2e2f6aac4062069cdd0fa5928d577b8404ce34131fab87bd4a7ae703`). It predates the current sources and was built against an older PyTorch, so it does not match the locked torch version; use the JIT build.

## Web

`web/` is a browser front end for single-image synthesis: style image, any number of guide pairs and the synthesis parameters; everything runs locally on the GPU through WebGPU (current Chrome/Edge; Firefox/Safari with WebGPU enabled). Build and serve it:

```powershell
.\web\build.ps1                     # needs the local toolchain and wasm-bindgen-cli 0.2.130
python -m http.server -d web 8000   # then open http://localhost:8000
```

The facestyle example (1000×810, three guides) takes about 0.6 s per run in Chrome on an RTX 3060.

## Usage

ReEzSynth can be used in two ways: via the command line for project-based work, or as a Python library for scripting and integration.

### Method 1: Command-Line (YAML Projects)

This method is ideal for managing full video projects.

**1. Prepare Video Frames**:
Extract your source video into a sequence of frames (e.g., PNGs).

```bash
uv run reezsynth-prepare --video "path/to/your/video.mp4" --output "projects/my_project/content"
```

**2. Create Keyframes**:
Select one or more frames, copy them to a `style` directory, and paint over them.

**3. Configure Project**:
Copy `configs/example_project.yml` and edit the paths for your project.

```yaml
# projects/my_project/config.yml
project:
  name: "MyFirstProject"
  content_dir: "projects/my_project/content"
  style_path: "projects/my_project/style/00123.png" # Path to your painted keyframe
  style_indices: [123]                                # Frame number of your keyframe
  output_dir: "output/MyFirstProject"
  cache_dir: "cache/MyFirstProject"
...
```

**4. Run Synthesis**:

```bash
uv run reezsynth-video --config "configs/example_project.yml"
```

The stylized frames will be saved to your specified `output_dir`.

### Method 2: Python API (Scripting)

This method is perfect for direct scripting, experimentation, and integrating ReEzSynth into other tools.

**Video Synthesis Example:**

```python
from ezsynth.api import Ezsynth, RunConfig

# 1. Configure parameters
config = RunConfig(
    pyramid_levels=5,
    uniformity=4000.0,
    use_sparse_feature_guide=True,
    use_temporal_nnf_propagation=True,
)

# 2. Initialize the synthesizer
synth = Ezsynth(
    content_dir="projects/my_project/content",
    style_paths=["projects/my_project/style/00123.png"],
    style_indices=[123],
    output_dir="output/api_video_output",
    config=config,
)

# 3. Run and get the results
final_frames = synth.run()
# Frames are saved automatically to output_dir and also returned as a list.
```

**Single Image Synthesis Example:**

```python
from ezsynth.api import ImageSynth, RunConfig, load_guide
from ezsynth.utils.io_utils import write_image

# 1. Configure parameters
config = RunConfig(patch_size=7, uniformity=4000)

# 2. Initialize the synthesizer
synth = ImageSynth(style_image="path/to/style.png", config=config)

# 3. Run synthesis by providing a list of guides
stylized_image, error_map = synth.run(
    guides=[
        load_guide("source_guide1.png", "target_guide1.png", weight=2.0),
        load_guide("source_guide2.png", "target_guide2.png", weight=1.5),
    ]
)

# 4. Save the output
write_image("output/stylized.png", stylized_image)
```

## Configuration Details

All settings are managed via a central YAML configuration file (`configs/default.yml`) or the `RunConfig` class in the API.

### Key `blending` settings

- `poisson_solver`: Select the solver for the blending step.
  - `"lsqr"` / `"lsmr"`: High-quality iterative solvers. Good defaults.
  - `"cg"`: Conjugate Gradient solver. Can be faster.
  - `"amg"`: **Fastest high-quality CPU option for large images.** Requires `pip install pyamg`.
  - `"seamless"`: Fast alternative using OpenCV's `seamlessClone`.
  - `"disabled"`: Fastest option. Skips blending, resulting in a hard cut.

### Key `ebsynth_params`

- `uniformity`: High values enforce texture consistency. Good default is `3500`.
- `patch_size`: Size of patches to match. Must be odd. `7` is a good balance.
- `search_pruning_threshold`: A speed optimization. Skips random search for patches with low error. A value of `50.0` is a good starting point.
- `*_weight`: Controls the influence of different guides (edges, color, position, etc.).

### Key `pipeline` settings

- `use_temporal_nnf_propagation`: Set to `true` to improve temporal stability.
- `use_sparse_feature_guide`: Set to `true` to track features and reduce sliding/jitter.

## How It Works

ReEzSynth operates in several stages:

1. **Pre-computation**: Optical flow fields and edge maps are calculated for the entire content sequence and cached to disk.
2. **Synthesis Pass**: The video is broken into sequences based on keyframe placement. For each sequence, the engine propagates the style from the keyframe to subsequent frames using the core patch-matching algorithm. This can be done forward or backward in time.
3. **Blending**: For sequences between two keyframes, the forward and reverse passes are intelligently blended. ReEzSynth uses a robust Poisson blending solver for this, ensuring a smooth and stable final output by minimizing temporal artifacts.

## Troubleshooting

- **`ebsynth_torch` not found**: the JIT build of the C++/CUDA extension failed. Check the compiler (and, for CUDA kernels, CUDA Toolkit) prerequisites and the build log, or use the `torch`/`taichi` backends, which need no extension.
- **CUDA Out of Memory**: the run exceeded its GPU memory budget (see Installation → GPU memory budget). Close other GPU applications, reduce content resolution, or raise the budget with `EZSYNTH_GPU_MEMORY_LIMIT_MB` / `EZSYNTH_GPU_MEMORY_RESERVE_MB`.
- **Visual Artifacts (Jitter/Flicker)**: Enable `use_temporal_nnf_propagation` and `use_sparse_feature_guide` in your config. These new features are specifically designed to solve these common problems. Adding more keyframes can also help.

## Development

```bash
uv sync --extra taichi --extra pyamg   # dev tools (pytest, ruff, pre-commit) are in the default dev group
uv run pre-commit install              # ruff check/format + uv lock on commit
uv run pytest                          # CPU-safe test suite
uv run ruff check . && uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs the same lint and tests on Linux. Backend notes live in `docs/` (`backend-benchmarks.md`, `taichi-porting-guide.md`).

## License

MIT (see `LICENSE`). Bundled third-party code (RAFT, NeuFlow v2, FastBlend, FaceBlit) keeps its own license; see `NOTICE`.

## Acknowledgements

- **jamriska** for the original EbSynth C++/CUDA source code: <https://github.com/jamriska/ebsynth>
- **Trentonom0r3** for the original Python API: <https://github.com/Trentonom0r3/Ezsynth>
- **Zachary Teed & Jia Deng** for the RAFT optical flow model: <https://github.com/princeton-vl/RAFT>
- **Zhiyong Zhang et al.** for the NeuFlow v2 optical flow model: <https://github.com/neufieldrobotics/NeuFlow_v2>
- Normalized Cross-Correlation (NCC) cost function used for patch matching was adapted from the implementation in the **ACMH** project, which provides excellent robustness for challenging matching tasks. <https://github.com/GhiXu/ACMH>
- **johnBuffer** for the Stable Index Vector implementation. <https://github.com/johnBuffer/StableIndexVector>
- **Taichi Lang** <https://github.com/taichi-dev/taichi>
- **RongxuanTan** for Bilateral weighting and Sparse Sampling. <https://github.com/RongxuanTan/MP-MVS>

- **Gemini 2.5 Pro** (via Google AI Studio) for assistance with coding and documentation.

```bibtex
@misc{Jamriska2018,
  author = {Jamriska, Ondrej},
  title = {Ebsynth: Fast Example-based Image Synthesis and Style Transfer},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jamriska/ebsynth}},
}
```

```bibtex
@misc{teed2020raftrecurrentallpairsfield,
      title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow}, 
      author={Zachary Teed and Jia Deng},
      year={2020},
      eprint={2003.12039},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2003.12039}, 
}
```

```bibtex
@misc{zhang2025neuflowv2pushhighefficiency,
      title={NeuFlow v2: Push High-Efficiency Optical Flow To the Limit}, 
      author={Zhiyong Zhang and Aniket Gupta and Huaizu Jiang and Hanumant Singh},
      year={2025},
      eprint={2408.10161},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10161}, 
}
```

```bibtex
@article{Xu2019ACMM,  
  title={Multi-Scale Geometric Consistency Guided Multi-View Stereo}, 
  author={Xu, Qingshan and Tao, Wenbing}, 
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
@article{Xu2020ACMP,  
  title={Planar Prior Assisted PatchMatch Multi-View Stereo}, 
  author={Xu, Qingshan and Tao, Wenbing}, 
  journal={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```
