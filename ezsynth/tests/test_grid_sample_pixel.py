import torch
import torch.nn.functional as F

from ezsynth.torch_ops.grid_sample_bilinear_pixel import (
    grid_sample_bilinear_pixel_coords,
)


def test_regular_sizes_match_align_corners_grid_sample():
    gen = torch.Generator().manual_seed(0)
    img = torch.rand((2, 3, 5, 7), generator=gen)
    coords = torch.rand((2, 4, 6, 2), generator=gen) * 9 - 1
    grid = torch.cat([2 * coords[..., :1] / 6 - 1, 2 * coords[..., 1:] / 4 - 1], dim=-1)
    expected = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    sampled, mask = grid_sample_bilinear_pixel_coords(img, coords, return_oob_mask=True)
    assert torch.equal(sampled, expected)
    assert mask.sum() > 0


def test_one_pixel_axis_samples_with_zero_padding():
    img = torch.tensor([10.0, 20.0, 30.0]).view(1, 1, 3, 1)  # width 1
    xs = torch.tensor([0.0, 0.5, -0.25, 1.0, 3.0])
    coords = torch.stack([xs, torch.ones_like(xs)], dim=-1).view(1, 1, 5, 2)
    sampled, mask = grid_sample_bilinear_pixel_coords(img, coords, return_oob_mask=True)
    expected = 20.0 * (1 - xs.abs()).clamp(min=0)
    torch.testing.assert_close(sampled.view(-1), expected)
    assert torch.equal(mask, torch.zeros_like(mask))


def test_one_by_one_image_is_finite():
    img = torch.full((1, 2, 1, 1), 4.0)
    coords = torch.tensor([[[[0.0, 0.0], [0.5, 0.5], [2.0, 0.0]]]])
    sampled = grid_sample_bilinear_pixel_coords(img, coords)
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled[0, 0, 0], torch.tensor([4.0, 1.0, 0.0]))
