# FastBlend

CUDA Extension for FastBlend. Based on the original implementation by AInseven and Artiprocher.

Included both CUDA and CuPy implementations (original).

## Example output

## Usage

### Quick Start

#### Frame-by-Frame Blending
```bash
python fastblend_standalone.py
    --content_dir <content_dir> # original content frames
    --style_dir <style_dir> # stylized frames
    --output_dir <output_dir> # output directory
```

#### Keyframe Interpolation
```bash
python fastblend_standalone_interpolate.py
    --frames_dir <frames_dir> # all original frames
    --keyframes_dir <keyframes_dir> # stylized keyframes (every Nth frame)
    --output_dir <output_dir> # output directory
    --keyframe_interval <N> # interval between keyframes
```

### Parameters

> Taken from [Artiprocher/sd-webui-fastblend](https://github.com/Artiprocher/sd-webui-fastblend)

FastBlend offers three accuracy modes with different trade-offs between speed, memory usage, and quality:

| Mode | Accuracy | Window Size | Batch Size | Min Patch Size | Iterations | Guide Weight | Description |
|------|----------|-------------|------------|----------------|------------|--------------|-------------|
| **Fast** (1) | ■ | 5 | 32 | 7 | 3 | 5.0 | Fast processing with minimal memory usage. Good for quick results. |
| **Balanced** (2) | ■■ | 15 | 16 | 5 | 5 | 10.0 | Balanced performance for most use cases. Recommended default. |
| **Accurate** (3) | ■■■ | 25 | 8 | 3 | 8 | 15.0 | Highest quality but slower and more memory intensive. |

#### Parameter Explanations

- **accuracy**: Quality level (1=Fast, 2=Balanced, 3=Accurate). Higher values produce better results but require more time and memory.
- **window_size**: Temporal window size for blending. Larger values create smoother videos but may introduce blurring. Controls how many frames are considered when blending each frame.
- **batch_size**: Processing batch size. Larger values speed up processing but require more VRAM. Reduce if you encounter memory errors.
- **minimum_patch_size**: Minimum patch size for matching (odd numbers only). Larger values are more stable but may miss fine details. Must be odd (3, 5, 7, etc.).
- **num_iter**: Number of patch matching iterations. Higher values improve quality but increase processing time.
- **guide_weight**: Controls how much motion information from guide frames is applied to style frames. Higher values preserve more motion details.

#### Performance Characteristics

- **Time Complexity** (approximate): `log(window_size) × minimum_patch_size² × num_iter × video_length × resolution`
- **Memory Usage**: Scales with `batch_size × window_size × frame_resolution`
- **Quality vs Speed**: Fast mode is ~3x faster than Accurate mode, with ~70% less memory usage

#### Recommended Settings

- **Quick preview**: accuracy=1, window_size=5, batch_size=16
- **Production video**: accuracy=2, window_size=15, batch_size=8-16 (adjust based on VRAM)
- **High quality**: accuracy=3, window_size=25, batch_size=4-8 (for high-end GPUs)
- **Low memory**: Reduce batch_size and window_size proportionally

### Keyframe Interpolation

FastBlend also supports keyframe interpolation, where you provide stylized frames at regular intervals (keyframes) and FastBlend interpolates the missing frames between them. This is useful for:

- Reducing computation by only stylizing every Nth frame
- Creating smooth transitions between manually edited keyframes
- Long video processing with limited compute resources

#### How It Works

1. **Prepare keyframes**: Create stylized versions of frames at regular intervals (e.g., every 10th frame)
2. **Run interpolation**: FastBlend uses patch matching to interpolate between keyframes
3. **Get smooth video**: All frames are filled in with temporally coherent results

#### Interpolation Parameters

- **Keyframe interval**: How often keyframes appear (e.g., every 10 frames)
- **Patch size**: Larger values (15+) recommended for interpolation stability
- **Batch size**: Smaller values (8-16) work better for interpolation

#### Example Workflow

```bash
# 1. Extract all frames from video
# 2. Stylize every Nth frame (keyframes) with your preferred method
# 3. Run interpolation to fill in the gaps
python fastblend_standalone_interpolate.py \
    --frames_dir ./all_frames \
    --keyframes_dir ./stylized_keyframes \
    --output_dir ./output \
    --keyframe_interval 10 \
    --accuracy 2
```

When `--keyframe_interval N` is specified, FastBlend uses every Nth frame from the keyframes directory as keyframes for interpolation. For example, with interval 2, it uses frames 0, 2, 4, 6, 8, 10, etc. from the keyframes directory.

### API

#### Main Functions

```python
from fastblend import smooth_video, create_config, check_frames_compatibility, interpolate_video

# Basic frame blending
smoothed_frames = smooth_video(guide_frames, style_frames)

# Keyframe interpolation
keyframe_indices = [0, 10, 20, 30]  # keyframes at frames 0, 10, 20, 30
interpolated_frames = interpolate_video(all_frames, keyframe_frames, keyframe_indices)

# Advanced usage with custom configuration
config = create_config(accuracy=2, window_size=10, batch_size=8)
smoothed_frames = smooth_video(guide_frames, style_frames, config=config)

# Check frame compatibility
warning = check_frames_compatibility(guide_frames, style_frames)
if warning:
    print(f"Warning: {warning}")
```

#### `smooth_video()`

Apply FastBlend post-processing to stylized video frames.

**Parameters:**

- `frames_guide`: List of guide frames (content frames) as numpy arrays (H, W, 3) uint8
- `frames_style`: List of stylized frames to smooth as numpy arrays (H, W, 3) uint8
- `accuracy`: Accuracy level (1=Fast, 2=Balanced, 3=Accurate). Default: 2
- `window_size`: Temporal window size for blending. Uses default if None
- `batch_size`: Batch size for processing. Uses default if None
- `progress_callback`: Optional callback function called with (current_frame, total_frames)
- `backend`: Backend to use ("auto", "cuda", "cupy"). Default: "auto"
- `config`: Optional FastBlendConfig. If provided, other parameters are ignored

**Returns:** List of smoothed frames as numpy arrays (H, W, 3) uint8

#### `create_config()`

Create a FastBlend configuration with custom parameters.

**Parameters:**

- `accuracy`: Base accuracy level (1=Fast, 2=Balanced, 3=Accurate). Default: 2
- `window_size`: Temporal window size for blending
- `batch_size`: Batch size for processing
- `minimum_patch_size`: Minimum patch size for matching
- `num_iter`: Number of patch matching iterations
- `guide_weight`: Weight for guide matching
- `backend`: Backend to use ("auto", "cuda", "cupy"). Default: "auto"
- `gpu_id`: GPU device ID. Default: 0

**Returns:** FastBlendConfig instance

#### `interpolate_video()`

Interpolate frames between keyframes using FastBlend.

**Parameters:**
- `frames_guide`: List of all guide frames (content frames) as numpy arrays (H, W, 3) uint8
- `keyframes_style`: List of stylized keyframe frames as numpy arrays (H, W, 3) uint8
- `keyframe_indices`: Indices in frames_guide that correspond to keyframes
- `accuracy`: Accuracy level (1=Fast, 2=Balanced, 3=Accurate). Default: 2
- `window_size`: Temporal window size for blending. Uses default if None
- `batch_size`: Batch size for processing. Uses default if None
- `progress_callback`: Optional callback function called with (current_frame, total_frames)
- `backend`: Backend to use ("auto", "cuda", "cupy"). Default: "auto"
- `config`: Optional FastBlendConfig. If provided, other parameters are ignored

**Returns:** List of interpolated frames as numpy arrays (H, W, 3) uint8

#### `check_frames_compatibility()`

Check compatibility of guide and style frames.

**Parameters:**
- `frames_guide`: Guide frames
- `frames_style`: Style frames

**Returns:** Warning message if issues found, empty string otherwise

#### Configuration Class

```python
from fastblend import FastBlendConfig, get_default_config

# Get default configuration
config = get_default_config(accuracy=2)  # 1=Fast, 2=Balanced, 3=Accurate

# Create custom configuration
config = FastBlendConfig(
    accuracy=2,
    window_size=15,
    batch_size=16,
    minimum_patch_size=5,
    num_iter=5,
    guide_weight=10.0,
    backend="cuda",
    gpu_id=0
)
```

#### Runner Class

```python
from fastblend import FastBlendRunner

# Create runner with configuration
runner = FastBlendRunner(config)

# Process frames
smoothed_frames = runner.run(guide_frames, style_frames, progress_callback, backend)
```

## References

- [FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier](https://arxiv.org/abs/2311.09265)
- [ComfyUI-fastblend](https://github.com/AInseven/ComfyUI-fastblend)
- [sd-webui-fastblend](https://github.com/Artiprocher/sd-webui-fastblend)

```bibtex
@article{duan2023fastblend,
  title={FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier},
  author={Duan, Zhongjie and Wang, Chengyu and Chen, Cen and Qian, Weining and Huang, Jun and Jin, Mingyi},
  journal={arXiv preprint arXiv:2311.09265},
  year={2023}
}
```
