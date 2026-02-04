import platform
from typing import Optional

import numpy as np
import taichi as ti
import torch


def get_ti_arch():
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return ti.metal
    elif torch.cuda.is_available():
        return ti.cuda
    return ti.cpu


_ti_initialized = False


def ensure_ti_init():
    global _ti_initialized
    if not _ti_initialized:
        arch = get_ti_arch()
        ti.init(arch=arch, log_level=ti.INFO, random_seed=42)
        print(f"[Taichi FastBlend] Initialized with arch: {arch}")
        _ti_initialized = True


@ti.data_oriented
class FastBlendTaichiBackend:
    def __init__(self):
        ensure_ti_init()
        self.device = "cpu"
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

    @ti.kernel
    def remap_kernel(
        self,
        height: int,
        width: int,
        channel: int,
        patch_size: int,
        pad_size: int,
        source_style: ti.types.ndarray(),
        nnf: ti.types.ndarray(),
        target_style: ti.types.ndarray(),
    ):
        r = (patch_size - 1) // 2
        for b, x, y in ti.ndrange(source_style.shape[0], height, width):
            num = 0.0

            min_px = -x if x < r else -r
            max_px = height - 1 - x if x + r > height - 1 else r
            min_py = -y if y < r else -r
            max_py = width - 1 - y if y + r > width - 1 else r

            for px, py in ti.ndrange((min_px, max_px + 1), (min_py, max_py + 1)):
                nx, ny = x + px, y + py

                sy_match = nnf[b, nx, ny, 0]
                sx_match = nnf[b, nx, ny, 1]

                sy = sy_match - px
                sx = sx_match - py

                if 0 <= sy < height and 0 <= sx < width:
                    num += 1.0
                    for c in range(channel):
                        target_style[b, x + pad_size, y + pad_size, c] += source_style[
                            b, sy + pad_size, sx + pad_size, c
                        ]

            if num > 0:
                for c in range(channel):
                    target_style[b, x + pad_size, y + pad_size, c] /= num

    @ti.kernel
    def patch_error_kernel(
        self,
        height: int,
        width: int,
        channel: int,
        patch_size: int,
        pad_size: int,
        source: ti.types.ndarray(),
        nnf: ti.types.ndarray(),
        target: ti.types.ndarray(),
        error: ti.types.ndarray(),
    ):
        r = (patch_size - 1) // 2
        for b, y, x in ti.ndrange(source.shape[0], height, width):
            sy_match = nnf[b, y, x, 0]
            sx_match = nnf[b, y, x, 1]

            e = 0.0
            for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
                typ, txp = y + pad_size + py, x + pad_size + px
                syp, sxp = sy_match + pad_size + py, sx_match + pad_size + px

                for c in range(channel):
                    diff = target[b, typ, txp, c] - source[b, syp, sxp, c]
                    e += diff * diff
            error[b, y, x] = e

    @ti.kernel
    def pairwise_patch_error_kernel(
        self,
        height: int,
        width: int,
        channel: int,
        patch_size: int,
        pad_size: int,
        source_a: ti.types.ndarray(),
        nnf_a: ti.types.ndarray(),
        source_b: ti.types.ndarray(),
        nnf_b: ti.types.ndarray(),
        error: ti.types.ndarray(),
    ):
        r = (patch_size - 1) // 2
        for b, y, x in ti.ndrange(source_a.shape[0], height, width):
            sy_a = nnf_a[b, y, x, 0]
            sx_a = nnf_a[b, y, x, 1]
            sy_b = nnf_b[b, y, x, 0]
            sx_b = nnf_b[b, y, x, 1]

            e = 0.0
            for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
                syp_a, sxp_a = sy_a + pad_size + py, sx_a + pad_size + px
                syp_b, sxp_b = sy_b + pad_size + py, sx_b + pad_size + px

                for c in range(channel):
                    diff = source_a[b, syp_a, sxp_a, c] - source_b[b, syp_b, sxp_b, c]
                    e += diff * diff
            error[b, y, x] = e


class PatchMatcherTaichi:
    def __init__(
        self,
        height,
        width,
        channel,
        minimum_patch_size,
        threads_per_block=16,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        random_search_steps=3,
        random_search_range=4,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
    ):
        self.backend = FastBlendTaichiBackend()
        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.random_search_steps = random_search_steps
        self.random_search_range = random_search_range
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        self.patch_size_list = [minimum_patch_size + i * 2 for i in range(num_iter)][
            ::-1
        ]
        self.pad_size = self.patch_size_list[0] // 2
        self.patch_size = self.patch_size_list[0]

    def pad_image(self, image):
        pad_size = self.pad_size
        return (
            torch.nn.functional.pad(
                image.permute(0, 3, 1, 2),
                (pad_size, pad_size, pad_size, pad_size),
                mode="reflect",
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def unpad_image(self, image):
        pad_size = self.pad_size
        return image[:, pad_size:-pad_size, pad_size:-pad_size, :].contiguous()

    def apply_nnf_to_image(self, nnf, source):
        batch_size = source.shape[0]
        target = torch.zeros(
            (
                batch_size,
                self.height + self.pad_size * 2,
                self.width + self.pad_size * 2,
                self.channel,
            ),
            dtype=torch.float32,
            device=source.device,
        )
        self.backend.remap_kernel(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            source,
            nnf,
            target,
        )
        return target

    def get_patch_error(self, source, nnf, target):
        batch_size = source.shape[0]
        error = torch.zeros(
            (batch_size, self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        self.backend.patch_error_kernel(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            source,
            nnf,
            target,
            error,
        )
        return error

    def get_pairwise_patch_error(self, source, nnf):
        batch_size = source.shape[0] // 2
        source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
        source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
        error = torch.zeros(
            (batch_size, self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        self.backend.pairwise_patch_error_kernel(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            source_a,
            nnf_a,
            source_b,
            nnf_b,
            error,
        )
        error = error.repeat_interleave(2, dim=0)
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        if self.use_mean_target_style:
            target_style_mapped = self.apply_nnf_to_image(nnf, source_style)
            target_style_mapped = target_style_mapped.mean(dim=0, keepdim=True)
            target_style_base = target_style_mapped.repeat(
                source_guide.shape[0], 1, 1, 1
            )
        else:
            target_style_base = target_style

        if self.use_pairwise_patch_error:
            error_style = self.get_pairwise_patch_error(source_style, nnf)
        else:
            error_style = self.get_patch_error(source_style, nnf, target_style_base)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        nnf[..., 0] = torch.clamp(nnf[..., 0], 0, self.height - 1)
        nnf[..., 1] = torch.clamp(nnf[..., 1], 0, self.width - 1)
        return nnf

    def neighboor_step(self, nnf, d):
        if d == 0:
            upd_nnf = torch.cat([nnf[:, :1, :, :], nnf[:, :-1, :, :]], dim=1)
            upd_nnf[:, :, :, 0] += 1
        elif d == 1:
            upd_nnf = torch.cat([nnf[:, :, :1, :], nnf[:, :, :-1, :]], dim=2)
            upd_nnf[:, :, :, 1] += 1
        elif d == 2:
            upd_nnf = torch.cat([nnf[:, 1:, :, :], nnf[:, -1:, :, :]], dim=1)
            upd_nnf[:, :, :, 0] -= 1
        elif d == 3:
            upd_nnf = torch.cat([nnf[:, :, 1:, :], nnf[:, :, -1:, :]], dim=2)
            upd_nnf[:, :, :, 1] -= 1
        return self.clamp_bound(upd_nnf)

    def shift_nnf(self, nnf, d):
        if d > 0:
            d = min(nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[d:], nnf[-1:].repeat(d, 1, 1, 1)], dim=0)
        else:
            d = max(-nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[:1].repeat(-d, 1, 1, 1), nnf[:-d]], dim=0)
        return upd_nnf

    def track_step(self, nnf, d):
        if self.use_pairwise_patch_error:
            upd_nnf = torch.zeros_like(nnf)
            upd_nnf[0::2] = self.shift_nnf(nnf[0::2], d)
            upd_nnf[1::2] = self.shift_nnf(nnf[1::2], d)
        else:
            upd_nnf = self.shift_nnf(nnf, d)
        return upd_nnf

    def update(
        self, source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf
    ):
        upd_err = self.get_error(
            source_guide, target_guide, source_style, target_style, upd_nnf
        )
        upd_idx = upd_err < err
        nnf[upd_idx] = upd_nnf[upd_idx]
        err[upd_idx] = upd_err[upd_idx]
        return nnf, err

    def propagation(
        self, source_guide, target_guide, source_style, target_style, nnf, err
    ):
        directions = torch.randperm(4, device=nnf.device)
        for d in directions:
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, err = self.update(
                source_guide,
                target_guide,
                source_style,
                target_style,
                nnf,
                err,
                upd_nnf,
            )
        return nnf, err

    def random_search(
        self, source_guide, target_guide, source_style, target_style, nnf, err
    ):
        batch_size = nnf.shape[0]
        for i in range(self.random_search_steps):
            step = torch.randint(
                -self.random_search_range,
                self.random_search_range + 1,
                (batch_size, self.height, self.width, 2),
                dtype=torch.int32,
                device=nnf.device,
            )
            upd_nnf = self.clamp_bound(nnf + step)
            nnf, err = self.update(
                source_guide,
                target_guide,
                source_style,
                target_style,
                nnf,
                err,
                upd_nnf,
            )
        return nnf, err

    def track(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for d in range(1, self.tracking_window_size + 1):
            upd_nnf = self.track_step(nnf, d)
            nnf, err = self.update(
                source_guide,
                target_guide,
                source_style,
                target_style,
                nnf,
                err,
                upd_nnf,
            )
            upd_nnf = self.track_step(nnf, -d)
            nnf, err = self.update(
                source_guide,
                target_guide,
                source_style,
                target_style,
                nnf,
                err,
                upd_nnf,
            )
        return nnf, err

    def iteration(
        self, source_guide, target_guide, source_style, target_style, nnf, err
    ):
        nnf, err = self.propagation(
            source_guide, target_guide, source_style, target_style, nnf, err
        )
        nnf, err = self.random_search(
            source_guide, target_guide, source_style, target_style, nnf, err
        )
        if self.tracking_window_size > 0:
            nnf, err = self.track(
                source_guide, target_guide, source_style, target_style, nnf, err
            )
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        source_guide_pad = self.pad_image(source_guide)
        target_guide_pad = self.pad_image(target_guide)
        source_style_pad = self.pad_image(source_style)

        for it in range(self.num_iter):
            self.patch_size = self.patch_size_list[it]
            target_style_pad = self.apply_nnf_to_image(nnf, source_style_pad)
            err = self.get_error(
                source_guide_pad,
                target_guide_pad,
                source_style_pad,
                target_style_pad,
                nnf,
            )
            nnf, err = self.iteration(
                source_guide_pad,
                target_guide_pad,
                source_style_pad,
                target_style_pad,
                nnf,
                err,
            )

        final_target_style = self.unpad_image(
            self.apply_nnf_to_image(nnf, source_style_pad)
        )
        return nnf, final_target_style


class PyramidPatchMatcherTaichi:
    def __init__(
        self,
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block=16,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity",
    ):
        maximum_patch_size = minimum_patch_size + (num_iter - 1) * 2
        self.pyramid_level = int(
            np.log2(min(image_height, image_width) / maximum_patch_size)
        )
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.initialize = initialize

        # Decide device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
        elif torch.backends.mps.is_available() and platform.system() == "Darwin":
            self.device = "mps"

        for level in range(self.pyramid_level):
            height = image_height // (2 ** (self.pyramid_level - 1 - level))
            width = image_width // (2 ** (self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(
                PatchMatcherTaichi(
                    height,
                    width,
                    channel,
                    minimum_patch_size=minimum_patch_size,
                    threads_per_block=threads_per_block,
                    num_iter=num_iter,
                    gpu_id=gpu_id,
                    guide_weight=guide_weight,
                    use_mean_target_style=use_mean_target_style,
                    use_pairwise_patch_error=use_pairwise_patch_error,
                    tracking_window_size=tracking_window_size,
                )
            )

    def resample_image(self, images, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        images = images.permute(0, 3, 1, 2)
        images_resample = torch.nn.functional.interpolate(
            images, size=(height, width), mode="area"
        )
        return images_resample.permute(0, 2, 3, 1).contiguous()

    def initialize_nnf(self, batch_size, height, width):
        if self.initialize == "random":
            nnf = torch.stack(
                [
                    torch.randint(
                        0,
                        height,
                        (batch_size, height, width),
                        device=self.device,
                        dtype=torch.int32,
                    ),
                    torch.randint(
                        0,
                        width,
                        (batch_size, height, width),
                        device=self.device,
                        dtype=torch.int32,
                    ),
                ],
                dim=3,
            )
        elif self.initialize == "identity":
            y_coords = (
                torch.arange(height, device=self.device, dtype=torch.int32)
                .view(height, 1)
                .repeat(1, width)
            )
            x_coords = (
                torch.arange(width, device=self.device, dtype=torch.int32)
                .view(1, width)
                .repeat(height, 1)
            )
            nnf = (
                torch.stack([y_coords, x_coords], dim=2)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
            )
        else:
            raise NotImplementedError()
        return nnf.contiguous()

    def update_nnf(self, nnf, level):
        nnf = nnf.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) * 2
        nnf[:, :, 1::2, 0] += 1
        nnf[:, 1::2, :, 1] += 1

        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[1] or width != nnf.shape[2]:
            nnf = nnf.permute(0, 3, 1, 2).float()
            nnf = torch.nn.functional.interpolate(
                nnf, size=(height, width), mode="bilinear", align_corners=False
            )
            nnf = nnf.permute(0, 2, 3, 1).int()
            nnf[..., 1] = torch.clamp(nnf[..., 1], 0, width - 1)
            nnf[..., 0] = torch.clamp(nnf[..., 0], 0, height - 1)
        return nnf.contiguous()

    def apply_nnf_to_image(self, nnf, image):
        image_pad = self.patch_matchers[-1].pad_image(image)
        image_remapped_pad = self.patch_matchers[-1].apply_nnf_to_image(nnf, image_pad)
        return self.patch_matchers[-1].unpad_image(image_remapped_pad)

    def estimate_nnf(self, source_guide, target_guide, source_style):
        if isinstance(source_guide, np.ndarray):
            source_guide = (
                torch.from_numpy(source_guide).float().to(self.device).contiguous()
            )
        if isinstance(target_guide, np.ndarray):
            target_guide = (
                torch.from_numpy(target_guide).float().to(self.device).contiguous()
            )
        if isinstance(source_style, np.ndarray):
            source_style = (
                torch.from_numpy(source_style).float().to(self.device).contiguous()
            )

        nnf = None
        target_style = None
        for level in range(self.pyramid_level):
            if level == 0:
                nnf = self.initialize_nnf(
                    source_guide.shape[0],
                    self.pyramid_heights[0],
                    self.pyramid_widths[0],
                )
            else:
                nnf = self.update_nnf(nnf, level)

            sg = self.resample_image(source_guide, level)
            tg = self.resample_image(target_guide, level)
            ss = self.resample_image(source_style, level)
            nnf, target_style = self.patch_matchers[level].estimate_nnf(sg, tg, ss, nnf)

        return nnf, target_style
