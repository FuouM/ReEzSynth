# FaceBlit (PyTorch)

Rewrite of [AnetaTexler/FaceBlit](https://github.com/AnetaTexler/FaceBlit) focused on face style transfer. This README covers the pure Python/PyTorch port used for research and integration into ReEzSynth.

## Examples

| Original | Style | Styled (CPU) | Styled (GPU) |
|:-:|:-:|:-:|:-:|
| <img src="examples/target2.png" height="250"> | <img src="examples/style_watercolorgirl.png" height="250"> | <img src="examples/output_refs/target2_stylized_pytorch_cpu.png" height="250"> | <img src="examples/output_refs/target2_stylized_pytorch_gpu.png" height="250"> |

## What's here

- `faceblit_pytorch`: Pure Python/PyTorch implementation with image stylization and style-asset precomputation.
- `examples/`: Sample assets and reference outputs used by tests.
- `models/`: Model files needed for style precompute and target landmark detection.

## Required assets

- Dlib 68-point landmark model (needed for style precompute and target landmark detection):  
  <https://github.com/AnetaTexler/FaceBlit/blob/master/VS/facemark_models/shape_predictor_68_face_landmarks.dat>
- Place the file at `FaceBlit/models/shape_predictor_68_face_landmarks.dat` (or point to it explicitly).
- Sample style/target images live in `FaceBlit/examples/` and are reused by the tests.

## Installation

- Windows users can grab prebuilt dlib wheels here if needed: <https://github.com/z-mahmud22/Dlib_Windows_Python3.x>

## Quickstart (PyTorch)

The port mirrors the FaceBlit flow: precompute style assets, prepare target guides, detect landmarks, then stylize. Try it out by running `python faceblit_pytorch/test_faceblit_pytorch.py`

## References

```bibtex
@Article{Texler21-I3D,
    author    = "Aneta Texler and Ond\v{r}ej Texler and Michal Ku\v{c}era and Menglei Chai and Daniel S\'{y}kora",
    title     = "FaceBlit: Instant Real-time Example-based Style Transfer to Facial Videos",
    journal   = "Proceedings of the ACM in Computer Graphics and Interactive Techniques",
    volume    = "4",
    number    = "1",
    year      = "2021",
}
```
