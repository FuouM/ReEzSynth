# ReEzSynth

<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD045 -->

ReEzSynth is a rewrite of [EbSynth](https://github.com/jamriska/ebsynth), the example-based synthesis tool that stylizes video from a few painted keyframes. It runs as a Python library and command line, as ComfyUI nodes, and in the browser. It also bundles FastBlend (video deflicker and keyframe interpolation) and FaceBlit (face style transfer).

Synthesis runs on one of several backends:

| Backend | Runs on | Needs | Notes |
|:-|:-|:-|:-|
| `wgpu` | Any GPU via Vulkan, DX12 or Metal | A prebuilt wheel (or Rust to build it) | Fastest; Rust + WebGPU; the same kernels run in the browser |
| `cuda` | NVIDIA GPUs | C++ compiler and CUDA Toolkit (JIT-built extension) | Native C++/CUDA extension |
| `taichi` | GPU or CPU | `taichi` (Python 3.10–3.12) | Default of the ComfyUI nodes |
| `torch` | Anything PyTorch runs on | Nothing extra | Pure PyTorch, slowest |

## Features

- **EbSynth parity**: any number of weighted guides (edges, colours, positional maps, …), modulation maps, and weighted or plain voting.
- **Video pipeline**: forward and reverse passes from each keyframe, blended between keyframes by error maps and histogram matching, then Poisson fusion (several solvers). Optical flow from RAFT, NeuFlow v2, OpenCV or TorchVision; edge guides from Classic, PAGE or PST.
- **Extra matching options**: SSD, NCC or SAD patch costs; PatchMatch, jump flooding or multi-candidate search; bilateral weights; float16/bfloat16 storage (torch and Taichi backends).
- **Temporal stability (experimental)**: temporal NNF propagation (each frame starts from the previous frame's matches) and sparse feature guides (tracked features pin the style to moving objects).
- **FastBlend**: blends a video stylized frame by frame against the original to remove flicker, or spreads stylized keyframes over every frame.
- **FaceBlit**: landmark-guided face style transfer.
- **Workflow**: YAML projects for the command line, a Python API (`Ezsynth`, `ImageSynth`), cached optical flow, and a GPU memory cap that leaves room for the desktop.

## Examples

| Face style | Stylit | Retarget |
|:-:|:-:|:-:|
| <img src="output_synth_api/facestyle_out.png" height="250"> | <img src="output_synth_api/stylit_out.png" height="250"> | <img src="output_synth_api/retarget_out.png" height="250"> |

<https://github.com/user-attachments/assets/0bdad993-e5ef-40bb-8c9a-47186e307fd5>

> Forward, Blend, Original and Reverse (clockwise)

<https://github.com/user-attachments/assets/770dd426-b708-4436-bc28-9302dad06ce5>

> Seamless, AMG, LSMR, LSQR, CG and Disabled blend methods (clockwise)

<https://github.com/user-attachments/assets/c909ebc9-b466-4907-9236-bf9e7447ea7c>

> Sum of squared differences (top) and normalized cross-correlation (bottom) patch costs

### Speed

All times below are on an RTX 3060 with a Ryzen 5 3600.

Single image, full pyramid (`run_img_synth.py` parameters):

| Backend | retarget 462×320 | stylit 912×1200 | facestyle 1000×810 |
|:-|-:|-:|-:|
| `wgpu` | 0.13 s | 0.97 s | 0.61 s |
| `cuda` | 0.17 s | 1.36 s | 0.85 s |
| `taichi` | 1.51 s | 8.84 s | 6.27 s |

Video, 100 frames at 960×544 with keyframes on the first and last frame:

- The bundled ComfyUI video workflow takes 2 min 43 s with `wgpu` and 12 min 39 s with `taichi`. The FastBlend workflow (98 frames) takes 125 s and 298 s.
- With `wgpu` the pipeline's image operations also run on the GPU. Blending 40 bidirectional frames takes 0.8 s instead of 13.4 s, and occlusion masks for 40 flow pairs take 0.3 s instead of 6.5 s.
- Poisson blending method and total pipeline time, including saving to disk. One EbSynth pass takes about 46 s and histogram blending about 22 s.

| Blend method | Pipeline time | Poisson solve |
|:-|-:|-:|
| none (forward or reverse pass only) | ~73 s | — |
| `disabled` | 147 s | — |
| `seamless` | 164 s | 0:17 |
| `cg` | 192 s | 0:52 |
| `amg` | 235 s | 1:31 |
| `lsmr` | 299 s | 2:36 |
| `lsqr` | 351 s | 3:33 |

## Installation

You need Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is used automatically when present; the `torch`, `taichi` and `wgpu` backends need no CUDA Toolkit.

```bash
git clone --filter=blob:none https://github.com/FuouM/ReEzSynth.git
cd ReEzSynth
uv sync                                   # core, with PyTorch for CUDA 12.6 (CPU/MPS build on macOS)
uv sync --extra taichi --extra pyamg      # + Taichi backend and the AMG Poisson solver
uv sync --all-extras                      # everything, incl. CuPy and FaceBlit (dlib needs CMake)
```

`--filter=blob:none` skips the large media in older history; current example media comes from Git LFS. To skip it as well, clone with `GIT_LFS_SKIP_SMUDGE=1` and fetch what you need later, for example `git lfs pull --include="examples/**"`.

Optional extras: `taichi`, `wgpu` (builds the Rust module from source), `cupy`, `pyamg`, `neuflow` (Hugging Face weight download), `faceblit`.

### `wgpu` backend

Every [GitHub Release](https://github.com/FuouM/ReEzSynth/releases) has prebuilt `ezsynth_wgpu` wheels: one per OS, for any Python ≥ 3.10 and any PyTorch.

```bash
pip install ezsynth_wgpu-<version>-cp310-abi3-<platform>.whl
```

To build it from source, you need a Rust toolchain:

- **Windows**: `setup_env.ps1` installs one into `.rust/`; load it with `. .\env.ps1`.
- **Elsewhere**: install one with [rustup](https://rustup.rs).

Then run `uv sync --extra wgpu` (or `pip install ./rust/ezsynth-wgpu`).

Select it with `backend: wgpu` or `--backend wgpu`; FastBlend `auto` prefers it after the native extension. To choose the graphics API and GPU, set `WGPU_BACKEND=vulkan|dx12|metal` and `WGPU_ADAPTER_NAME`.

With `wgpu` the video pipeline's flow warping, forward splatting, histogram blending and occlusion masks also run on WebGPU, so forward warping no longer needs Taichi. Warping and occlusion masks match OpenCV 5. Histogram blending converts to Lab in float, so a few pixels differ from OpenCV's 8-bit path (mean difference below one level).

### `cuda` / `cpu` native extension

The `ebsynth_torch` C++/CUDA extension (`ezsynth/ebsynth_extension/`) is JIT-compiled the first time the `cuda` or `cpu` backend is used, then cached by PyTorch.

- **Compiler**: C++20 (Visual Studio 2022 "Desktop development with C++" on Windows, GCC 10+ on Linux).
- **CUDA kernels**: CUDA Toolkit 12.4 or newer. The `.cu` files include no PyTorch headers and compile as C++17.
- **Target GPUs**: taken from `TORCH_CUDA_ARCH_LIST`, else the local GPU.

```bash
export EZSYNTH_NATIVE_ARCH=1         # opt in to -march=native / /arch:AVX2 host tuning
export FORCE_EBSYNTH_JIT_LOADER=1    # ignore an installed ebsynth_torch wheel and JIT-build
```

The Releases also hold an old prototype `ebsynth_torch-0.0.0-cp310-cp310-win_amd64.whl`. It doesn't match the current sources or PyTorch, so use the JIT build.

### GPU memory budget

Outside ComfyUI, each run caps PyTorch's CUDA allocator at 90% of the memory free at startup, always leaving 1.5 GiB for the desktop and browser. Exceeding the cap raises an out-of-memory error in ReEzSynth instead of exhausting the GPU. To change it:

- `EZSYNTH_GPU_MEMORY_LIMIT_MB`: absolute cap.
- `EZSYNTH_GPU_MEMORY_FRACTION`: fraction of the free memory.
- `EZSYNTH_GPU_MEMORY_RESERVE_MB`: memory kept free.
- `EZSYNTH_GPU_MEMORY_LIMIT=0`: disables the cap.

## ComfyUI

1. **Clone** into `ComfyUI/custom_nodes/ReEzSynth`, preferably without media: `GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/FuouM/ReEzSynth.git` (in Windows `cmd`, run `set GIT_LFS_SKIP_SMUDGE=1` first).
2. **Install the requirements** into ComfyUI's Python; torch and torchvision come from ComfyUI. Portable build: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ReEzSynth\requirements.txt`. The requirements include `taichi`, the nodes' default backend. Taichi has no wheels for Python 3.13+, so there use `wgpu` or `torch`.
3. **Optional, fastest**: install a `wgpu` wheel from the [Releases](https://github.com/FuouM/ReEzSynth/releases), for example `python_embeded\python.exe -m pip install ezsynth_wgpu-<version>-cp310-abi3-win_amd64.whl`, then pick `wgpu` in the config nodes. FastBlend `auto` uses it when it's installed.
4. **Optical flow models**: for the default RAFT engine, put the RAFT checkpoints in `models/raft/` (see `models/model_files.md`). TorchVision downloads its weights itself; OpenCV needs none.
5. **Restart ComfyUI.** The nodes appear under `ReEzSynth`: Guide, Guides Append, Image Synth Config, Image Synth, Video Style Keyframe(s), Video Style Keyframes Append, Video Synth Config, Video Synth, FastBlend Interpolate, FaceBlit.
6. **Example graphs** are in `workflows/`: image synthesis (`reezsynth_imgsynth_*.json`), `reezsynth_videosynth.json`, `reezsynth_fastblend.json` and `reezsynth_faceblit.json`. Drag one onto the canvas.

Some options need extra packages. A node that needs a missing one stops and shows the `pip install` command.

| Package | Needed for |
|:-|:-|
| `cupy-cuda12x` | FastBlend `cupy` backend |
| `phycv` | PAGE and PST edge methods |
| `pyamg` | `amg` Poisson solver |
| `huggingface-hub` | NeuFlow flow engine (checkpoints in `models/neuflow/`) |
| `face-alignment` | FaceBlit landmark detection when no landmarks are connected |
| `numba` | FaceBlit `numba` DFS mode |

The `cuda` backends JIT-build the native extension, which needs a C++ compiler and the CUDA Toolkit matching ComfyUI's PyTorch.

Inside ComfyUI the memory cap is off, since ComfyUI manages VRAM. The nodes:

- ask ComfyUI to offload its models before synthesis;
- report progress to the progress bar;
- stop at the next frame or pyramid level when you cancel.

The `torch` backend still sizes its tensors to keep `EZSYNTH_GPU_MEMORY_RESERVE_MB` (1.5 GiB by default) free.

## Browser app

`web/` is ReEzSynth in the browser, built with Solid, Vite and the [SoulWindow](https://github.com/FuouM/soulwindow) design system. It runs on the GPU through WebGPU in current Chrome or Edge, or Firefox/Safari with WebGPU enabled. Images and videos never leave the machine.

- **Image**: EbSynth on one image from a style image and any number of guide pairs.
- **Keyframes → video**: load a video or its frames, stylize a few frames in any tool and add them back as keyframes (a number in the file name places each one). Every other frame is then filled in by one of two methods:
  - **FastBlend interpolation**: fast.
  - **EbSynth**: the Python pipeline's guides, passes and pass blending, with NeuFlow optical flow; optionally deflickered with FastBlend afterwards.
- **Deflicker video**: FastBlend blending of a video stylized frame by frame, guided by the original.

Other details:

- **Files**: drop them on the input boxes or the monitor. The monitor has a before/after split (Compare, or the C key), and settings are kept between visits.
- **Video**: decoded and encoded in the browser with WebCodecs ([mediabunny](https://mediabunny.dev)). MP4, WebM or MOV go in; H.264 MP4 (or WebM) or a zip of PNGs come out.
- **Memory**: frames are held in memory at the working size (768 px on the longer side by default). For long or large videos, lower the working size or set a frame limit.

### Optical flow model

EbSynth video uses NeuFlow v2 through [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), loaded from jsDelivr on first use. Export the model once into `web/public/models/` (this needs the NeuFlow checkpoint in `models/neuflow/`), or choose an exported `.onnx` file in the app:

```bash
uv run --with onnx --with onnxscript python web/export_neuflow.py --size 960x544   # --fp16 halves it; needs WebGPU shader-f16
```

The model has a fixed input size. Frames up to 31 px smaller are padded; other sizes are resized and the flow rescaled.

On the cat clip (100 frames, 960×544, keyframes 0 and 99, histogram blending) the browser takes 2 min 52 s on an RTX 3060. Its frames are within 37–39 dB PSNR of the Python `wgpu` pipeline. Keyframes stay exactly as painted, whereas Python re-blends the first one.

### Build

This needs Node 22.18+, the `wasm32-unknown-unknown` Rust target and `wasm-bindgen-cli` 0.2.130.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
cd web
npm install
npm run wasm      # compiles rust/ebsynth-web into web/src/pkg (uses .rust/ from setup_env.ps1 when present)
npm run dev       # http://localhost:5173
npm run build     # static site in web/dist; serve it over https or localhost
npm run check     # TypeScript check and unit tests
```

## Usage

### Command line (YAML projects)

1. Extract the video into frames:

    ```bash
    uv run reezsynth-prepare --video "path/to/video.mp4" --output "projects/my_project/content"
    ```

2. Copy one or more frames into a `style` directory and paint over them.
3. Copy `configs/example_project.yml` and set your paths:

    ```yaml
    project:
      name: "MyFirstProject"
      content_dir: "projects/my_project/content"
      style_path: "projects/my_project/style/00123.png"  # the painted keyframe
      style_indices: [123]                               # its frame number
      output_dir: "output/MyFirstProject"
      cache_dir: "cache/MyFirstProject"
    ```

4. Run it; the stylized frames go to `output_dir`:

    ```bash
    uv run reezsynth-video --config "projects/my_project/config.yml"
    ```

`fastblend` (deflicker) and `fastblend-interpolate` (keyframe interpolation) are installed alongside. Each tool also runs as `python -m ezsynth.cli.<name>`.

### Python API

Video:

```python
from ezsynth.api import Ezsynth, RunConfig

config = RunConfig(
    pyramid_levels=5,
    uniformity=4000.0,
    use_sparse_feature_guide=True,
    use_temporal_nnf_propagation=True,
)
synth = Ezsynth(
    content_dir="projects/my_project/content",
    style_paths=["projects/my_project/style/00123.png"],
    style_indices=[123],
    output_dir="output/api_video_output",
    config=config,
)
frames = synth.run()  # also saved to output_dir
```

Single image:

```python
from ezsynth.api import ImageSynth, RunConfig, load_guide
from ezsynth.utils.io_utils import write_image

synth = ImageSynth(
    style_image="path/to/style.png", config=RunConfig(patch_size=7, uniformity=4000)
)
image, error = synth.run(
    guides=[
        load_guide("source_guide1.png", "target_guide1.png", weight=2.0),
        load_guide("source_guide2.png", "target_guide2.png", weight=1.5),
    ]
)
write_image("output/stylized.png", image)
```

## Configuration

Settings come from a YAML file (defaults in `configs/default.yml`) or from `RunConfig` in the API.

`ebsynth_params`:

- `uniformity`: higher values spread style patches more evenly; `3500` is a good default.
- `patch_size`: odd; `7` balances detail and speed.
- `search_pruning_threshold`: skips random search for patches whose error is already low; `50.0` is a good start.
- `cost_function` (`ssd`, `ncc`, `sad`) and `search_mode` (`patchmatch`, `jfa`, `multicand`): not supported by the native `cuda` backend.
- `edge_weight`, `image_weight`, `pos_weight`, `warp_weight`: guide weights (defaults 1, 6, 2, 0.5).

`blending.poisson_solver`:

- `lsqr`, `lsmr`: high-quality iterative solvers.
- `cg`: conjugate gradient; faster.
- `amg`: fastest high-quality CPU option for large frames; needs `pyamg`.
- `seamless`: OpenCV `seamlessClone`; fast.
- `disabled`: no Poisson fusion; only histogram blending.

`pipeline`:

- `use_temporal_nnf_propagation`: reuse the previous frame's matches to reduce flicker.
- `use_sparse_feature_guide`: track features to reduce sliding textures.

## How it works

1. **Precompute**: optical flow and edge maps for the whole sequence, cached on disk.
2. **Passes**: the keyframes split the video into sequences. Each frame is synthesized from the nearest keyframe's style, with these guides:
   - an edge map;
   - the frame itself;
   - a positional map moved along the flow;
   - the previous result warped onto the frame.

   Frames before the first keyframe run backwards from it, and frames after the last run forwards. Frames between two keyframes run both ways.
3. **Blend**: the two passes between keyframes are combined. Each pixel takes the pass with the lower patch error, and masks are carried along the flow. The passes are then histogram-matched in Lab and fused with a Poisson solve.

## Troubleshooting

- **`ebsynth_torch` not found**: the JIT build of the native extension failed. Check the compiler and CUDA Toolkit, or use the `wgpu`, `taichi` or `torch` backend.
- **CUDA out of memory**: the run exceeded its memory budget (see [GPU memory budget](#gpu-memory-budget)). Close other GPU applications, lower the resolution, or raise `EZSYNTH_GPU_MEMORY_LIMIT_MB` / lower `EZSYNTH_GPU_MEMORY_RESERVE_MB`.
- **Jitter or flicker**: add keyframes, enable `use_temporal_nnf_propagation` and `use_sparse_feature_guide`, or run FastBlend deflicker on the result.

## Development

```bash
uv sync --extra taichi --extra pyamg   # dev tools (pytest, ruff, pre-commit) are in the default dev group
uv run pre-commit install              # ruff check/format + uv lock on commit
uv run pytest                          # CPU-safe test suite
uv run ruff check . && uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs:

- ruff;
- the Python tests;
- rustfmt, clippy (native and wasm32) and WGSL validation;
- the web app's typecheck, tests and build;
- the `ezsynth-wgpu` wheels for Windows, Linux and macOS.

A pushed `v*` tag publishes the wheels to that tag's GitHub Release. Backend notes are in `docs/` (`backend-benchmarks.md`, `taichi-porting-guide.md`).

## License

MIT (see `LICENSE`). Bundled third-party code (RAFT, NeuFlow v2, FastBlend, FaceBlit) keeps its own license; see `NOTICE`.

## Acknowledgements

- **jamriska** for the original EbSynth C++/CUDA source code: <https://github.com/jamriska/ebsynth>
- **Trentonom0r3** for the original Python API: <https://github.com/Trentonom0r3/Ezsynth>
- **Zachary Teed & Jia Deng** for the RAFT optical flow model: <https://github.com/princeton-vl/RAFT>
- **Zhiyong Zhang et al.** for the NeuFlow v2 optical flow model: <https://github.com/neufieldrobotics/NeuFlow_v2>
- The Normalized Cross-Correlation (NCC) patch cost was adapted from the **ACMH** project: <https://github.com/GhiXu/ACMH>
- **johnBuffer** for the Stable Index Vector implementation: <https://github.com/johnBuffer/StableIndexVector>
- **Taichi Lang**: <https://github.com/taichi-dev/taichi>
- **RongxuanTan** for bilateral weighting and sparse sampling: <https://github.com/RongxuanTan/MP-MVS>
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
