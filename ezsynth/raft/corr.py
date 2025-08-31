import torch
import torch.nn.functional as F

from .utils import bilinear_sampler


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    """
    A memory-efficient correlation block implemented in pure PyTorch.
    This computes local correlations on-the-fly, avoiding the need to
    materialize the full all-pairs correlation matrix.
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmap1 = fmap1

        self.fmap2_pyramid = [fmap2]
        for _ in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.fmap2_pyramid.append(fmap2)

    def __call__(self, coords):
        r = self.radius
        B, _, H1, W1 = coords.shape
        corr_list = []

        dim = self.fmap1.shape[1]
        fmap1_flat = self.fmap1.view(B, dim, H1 * W1, 1)

        for i in range(self.num_levels):
            fmap2 = self.fmap2_pyramid[i]
            _, C, H2, W2 = fmap2.shape

            coords_i = (coords / 2**i).permute(0, 2, 3, 1)  # B, H1, W1, 2

            # Create sampling grid of shape B, H1*W1, (2r+1)^2, 2
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = (
                torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)
                .flip(-1)  # (y,x) -> (x,y)
                .view(1, 1, (2 * r + 1) ** 2, 2)
            )
            centroid_lvl = coords_i.reshape(B, H1 * W1, 1, 2)
            coords_lvl = centroid_lvl + delta

            # Normalize coordinates for grid_sample
            coords_lvl[..., 0] = 2 * coords_lvl[..., 0] / (W2 - 1) - 1
            coords_lvl[..., 1] = 2 * coords_lvl[..., 1] / (H2 - 1) - 1

            # Sample features from fmap2 at the grid locations
            fmap2_sampled = F.grid_sample(
                fmap2, coords_lvl, align_corners=True, padding_mode="zeros"
            )
            # -> B, C, H1*W1, (2r+1)^2

            # Compute dot product correlation
            corr = torch.sum(fmap1_flat * fmap2_sampled, dim=1)
            # -> B, H1*W1, (2r+1)^2

            corr = corr.view(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            # -> B, (2r+1)^2, H1, W1
            corr_list.append(corr)

        out = torch.cat(corr_list, dim=1)
        return out / torch.sqrt(
            torch.tensor(dim, dtype=torch.float32, device=out.device)
        )
