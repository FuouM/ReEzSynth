import time

import numpy as np
from faceblit_pytorch.src import ops
from faceblit_pytorch.src.api import FaceBlit as FaceBlitPytorch

from faceblit_taichi import FaceBlitTaichi


def test_faceblit_taichi():
    print("Testing FaceBlit Taichi vs Pytorch...")

    # Create dummies
    h, w = 256, 256
    # Use smooth style image (gradient)
    ys = np.linspace(0, 255, h)
    xs = np.linspace(0, 255, w)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    style_img = np.stack([np.zeros_like(grid_x), grid_y, grid_x], axis=-1).astype(
        np.uint8
    )

    # Use real-ish looking guides
    style_pos = ops.gradient_guide(w, h)
    # Use smooth app guide
    style_app = grid_x.astype(np.uint8)

    # 1. Test LUT computation parity
    fb_taichi = FaceBlitTaichi()

    start = time.time()
    fb_taichi.load_style(style_img, style_pos, style_app)
    taichi_lut_packed = fb_taichi.look_up_cube
    print(f"Taichi LUT time: {time.time() - start:.4f}s")

    start = time.time()
    torch_lut = ops.compute_look_up_cube_optimized(style_pos, style_app)
    print(f"Pytorch LUT time: {time.time() - start:.4f}s")

    # Unpack Taichi LUT
    tl_raw = taichi_lut_packed.cpu().numpy()
    taichi_lut_unpacked = np.zeros((256, 256, 256, 2), dtype=np.uint16)
    taichi_lut_unpacked[..., 0] = (tl_raw >> 10) & 0x3FF  # sx
    taichi_lut_unpacked[..., 1] = tl_raw & 0x3FF  # sy

    lut_diff = np.abs(taichi_lut_unpacked.astype(np.int32) - torch_lut.astype(np.int32))
    print(f"LUT Max diff: {lut_diff.max()}")
    print(f"LUT Mean diff: {lut_diff.mean():.4f}")

    if lut_diff.max() > 0:
        # Find where it first differs
        idx = np.unravel_index(np.argmax(lut_diff), lut_diff.shape)
        print(
            f"First diff at {idx}: Taichi={taichi_lut_unpacked[idx]}, Torch={torch_lut[idx]}"
        )

    # 2. Test Stylization parity
    target_pos = ops.gradient_guide(w, h)
    target_app = grid_x.astype(np.uint8)

    start = time.time()
    taichi_out = fb_taichi.stylize_with_guides(target_pos, target_app, patch_size=3)
    print(f"Taichi Stylize time: {time.time() - start:.4f}s")

    # Pytorch for comparison
    fb_torch = FaceBlitPytorch()
    fb_torch.style_image = style_img
    fb_torch.style_pos_guide = style_pos
    fb_torch.style_app_guide = style_app
    fb_torch.look_up_cube = torch_lut
    # Mock landmarks if needed, but style_blit_voting can be called directly

    start = time.time()
    torch_out_np = ops.style_blit_voting(
        style_pos,
        target_pos,
        style_app,
        target_app,
        torch_lut,
        style_img,
        patch_size=3,
        device="cpu",  # Use CPU to avoid device mismatch for comparison
    )
    print(f"Pytorch Stylize time: {time.time() - start:.4f}s")

    # taichi_out is on device, move to cpu
    taichi_out_np = taichi_out.cpu().numpy()

    diff = np.abs(taichi_out_np.astype(np.int16) - torch_out_np.astype(np.int16))
    print(f"Max pixel difference: {diff.max()}")
    print(f"Mean pixel difference: {diff.mean():.4f}")

    if diff.max() < 10:  # Allow some drift due to rounding/heuristics
        print("Success: Taichi vs Pytorch parity confirmed")
    else:
        print("Warning: significant difference detected")


if __name__ == "__main__":
    test_faceblit_taichi()
