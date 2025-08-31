# ReEzSynth: A PyTorch-Powered Remake of Ebsynth

ReEzSynth is a complete, from-the-ground-up rewrite and enhancement of the original Ebsynth video stylization tool. It leverages the power and flexibility of PyTorch and a custom CUDA extension to deliver high-performance, high-quality video-to-video synthesis, while maintaining full feature parity with the original's powerful artistic controls.

This project is designed for artists, researchers, and developers who need a robust, scriptable, and high-performance tool for temporal style transfer.

## Key Features

- **High-Performance CUDA Backend**: The core patch-matching and synthesis algorithm is implemented as a PyTorch C++/CUDA extension, ensuring maximum performance on modern GPUs.
- **Full Feature Parity with Original Ebsynth**:
  - **Multi-Guide Synthesis**: Use multiple guide layers (e.g., edges, colors, positional maps) with individual weights.
  - **Modulation Maps**: Spatially control the influence of guides for fine-grained artistic direction.
  - **Weighted & Plain Voting**: Choose between sharp, detail-preserving synthesis (`weighted`) or a softer, more painterly style (`plain`).
  - **Uniformity Control**: Prevent style "drift" and maintain texture consistency.
  - **Polishing Pass**: An optional final `extra_pass_3x3` to refine details, just like the original.
- **Advanced Temporal Stability**:
  - **Bidirectional Synthesis**: Processes video in both forward and reverse directions.
  - **Poisson Blending**: Seamlessly merges the forward and reverse passes using a high-quality Poisson solver (LSQR/LSMR) to eliminate jitter and temporal artifacts.
  - **High-Quality Optical Flow**: Integrates state-of-the-art optical flow models (e.g., RAFT) for accurate motion tracking.
- **Modern, Configurable Pipeline**:
  - **YAML-based Projects**: Easily define and manage complex projects with simple configuration files.
  - **Caching System**: Automatically caches expensive pre-computation steps (like optical flow and edge detection) to accelerate iterative workflows.
  - **Modular Engine Design**: Built with distinct engines for flow, edges, and synthesis, making it extensible.
- **Extensible and Scriptable**: As a Python library, ReEzSynth can be easily integrated into larger graphics pipelines and automated workflows.

## Installation

### Prerequisites

1. **NVIDIA GPU**: A CUDA-compatible NVIDIA GPU with Compute Capability 3.0 or higher is required.
2. **CUDA Toolkit**: You must have the NVIDIA CUDA Toolkit installed. This project has been tested with CUDA 11.x and 12.x.
3. **C++ Compiler**:
    - **Windows**: Visual Studio (2019 or newer) with the "Desktop development with C++" workload installed.
    - **Linux**: GCC/G++ (version 7 or newer).
4. **Python**: Python 3.8 or newer.
5. **Conda (Recommended)**: Using a Conda environment is highly recommended to manage dependencies.

### Setup

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/FuouM/ReEzSynth.git
    cd ReEzSynth
    ```

2. **Create and Activate a Conda Environment**:

    ```bash
    conda create -n reezsynth python=3.10
    conda activate reezsynth
    ```

3. **Install PyTorch with CUDA Support**:
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your specific CUDA version. For example:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4. **Install Dependencies and Build the CUDA Extension**:
    This single command will install all required Python packages and compile the `ebsynth_torch` C++/CUDA extension.

    ```bash
    pip install .
    ```

    If the installation is successful, you are ready to go!

## Quick Start

### 1. Prepare Your Video Frames

First, you need to extract your source video into a sequence of frames (e.g., PNG images). You can use the provided helper script:

```bash
python prepare_video.py --video "path/to/your/video.mp4" --output "projects/my_project/content"
```

### 2. Create Keyframes

Select one or more frames from your `content` directory, copy them to a `style` directory (e.g., `projects/my_project/style`), and paint over them in your desired style.

### 3. Configure Your Project

Copy the `configs/example_project.yml` file and edit it for your project.

**`projects/my_project/config.yml`:**

```yaml
project:
  name: "MyFirstProject"
  content_dir: "projects/my_project/content"
  style_path: "projects/my_project/style/00123.png" # Path to your painted keyframe
  style_indices: [123] # The frame number of your keyframe
  output_dir: "output/MyFirstProject"
  cache_dir: "cache/MyFirstProject"

# ... (keep other settings as default for now)
```

### 4. Run the Synthesis

Execute the pipeline using your configuration file:

```bash
python run.py --config "configs/example_project.yml"
```

The stylized frames will be saved to the `output_dir` specified in your config. You can then use software like FFMPEG or Adobe Premiere to compile them back into a video.

## Configuration Details

All settings are managed via a central YAML configuration file. See `configs/default.yml` for a comprehensive list of all available parameters and their descriptions.

### Key `ebsynth_params`

These parameters directly control the core synthesis algorithm and are crucial for artistic results. They mirror the original Ebsynth's settings.

- `uniformity`: High values enforce texture consistency. Good default is `3500`.
- `patch_size`: Size of patches to match. Must be odd. `7` is a good balance.
- `vote_mode`: `'weighted'` (default) for sharp results, or `'plain'` for a softer, more painterly effect.
- `extra_pass_3x3`: If `true`, performs a final high-detail polishing pass with 3x3 patches.
- `*_weight`: Controls the influence of different guides (edges, color, position, etc.).

## How It Works

ReEzSynth operates in several stages:

1. **Pre-computation**: Optical flow fields and edge maps are calculated for the entire content sequence and cached to disk.
2. **Synthesis Pass**: The video is broken into sequences based on the placement of keyframes. For each sequence, the engine propagates the style from the keyframe to subsequent frames.
    - **Forward Pass**: Renders frames from a keyframe to the end of a sequence.
    - **Reverse Pass**: Renders frames from a keyframe to the beginning of a sequence.
3. **Blending**: For sequences between two keyframes, the forward and reverse passes are intelligently blended together. This step is critical for eliminating temporal artifacts (like "wobble" or "jitter") that appear halfway between keyframes. ReEzSynth uses a robust Poisson blending solver for this, ensuring smooth and stable final output.

## Troubleshooting

- **`ebsynth_torch` not found**: This means the C++/CUDA extension did not build correctly. Ensure you have all prerequisites (CUDA Toolkit, C++ Compiler) and that your PyTorch version was installed with CUDA support. Re-run `pip install .` and check the build log for errors.
- **CUDA Out of Memory**: Video synthesis is memory-intensive. If you encounter OOM errors, try reducing the resolution of your content frames.
- **Visual Artifacts (Jitter/Flicker)**:
  - Ensure your optical flow model is appropriate for your content.
  - Try adding more keyframes in problematic areas.
  - Adjust the blending parameters or guide weights in your config file.

## Acknowledgements

**jamriska** for the original EbSynth C++/CUDA source code: <https://github.com/jamriska/ebsynth>.

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

**Trentonom0r3** for the original Python API <https://github.com/Trentonom0r3/Ezsynth>.

**Zachary Teed** and **Jia Deng** for the Optical Flow **RAFT** model: <https://github.com/princeton-vl/RAFT>.

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

**Zhiyong Zhang**, **Aniket Gupta**, **Huaizu Jiang** and **Hanumant Singh** for the Optical Flow **NeuFlow v2** model: <https://github.com/neufieldrobotics/NeuFlow_v2>.

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

And Gemini 2.5 Pro (via Google AI Studio) for most of the coding.
