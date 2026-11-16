import torch
import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        self._calculate_padding(mode)

    def _calculate_padding(self, mode):
        pad_ht = -self.ht % 8
        pad_wd = -self.wd % 8

        if mode == "sintel":
            self._pad = (
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            )
        else:
            self._pad = (pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht)

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        return x[
            ...,
            self._pad[2] : self.ht + self._pad[2],
            self._pad[0] : self.wd + self._pad[0],
        ]


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
