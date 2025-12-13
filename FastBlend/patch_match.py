from typing import Optional

import cv2
import numpy as np
import torch

try:
    from .fastblend_extension import (
        is_available,
        pairwise_patch_error,
        patch_error,
        remap,
    )
except ImportError:
    # Try cupy-based implementation as fallback
    try:
        from .cupy_patch_match import PatchMatcherCupy, PyramidPatchMatcherCupy

        cupy_available = True
    except ImportError:
        cupy_available = False

    if cupy_available:

        def is_available():
            return True

        # Create wrapper functions that delegate to cupy implementation
        def remap(source, nnf, patch_size, pad_size):
            # This would need to be implemented as a wrapper around the cupy patch matcher
            # For now, we'll mark cupy as available but the actual functions will need
            # to be called through the patch matcher classes
            raise NotImplementedError("Use cupy patch matcher classes directly")

        def patch_error(source, nnf, target, patch_size, pad_size):
            raise NotImplementedError("Use cupy patch matcher classes directly")

        def pairwise_patch_error(
            source_a, nnf_a, source_b, nnf_b, patch_size, pad_size
        ):
            raise NotImplementedError("Use cupy patch matcher classes directly")
    else:
        # Fallback if neither CUDA extension nor cupy available
        is_available = lambda: False
        remap = None
        patch_error = None
        pairwise_patch_error = None


# Factory functions for different backends
def create_patch_matcher_cuda(
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
    """Create CUDA patch matcher"""
    if not is_available():
        raise RuntimeError("CUDA extension not available")
    return PatchMatcherCUDA(
        height,
        width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        random_search_steps,
        random_search_range,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
    )


def create_patch_matcher_cupy(
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
    """Create CuPy patch matcher"""
    try:
        from .cupy_patch_match import PatchMatcherCupy

        return PatchMatcherCupy(
            height,
            width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            random_search_steps,
            random_search_range,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
        )
    except ImportError:
        raise RuntimeError("CuPy not available for FastBlend")


def create_pyramid_patch_matcher_cuda(
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
    """Create CUDA pyramid patch matcher"""
    if not is_available():
        raise RuntimeError("CUDA extension not available")
    return PyramidPatchMatcherCUDA(
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
        initialize,
    )


def create_pyramid_patch_matcher_cupy(
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
    """Create CuPy pyramid patch matcher"""
    try:
        from .cupy_patch_match import PyramidPatchMatcherCupy

        return PyramidPatchMatcherCupy(
            image_height,
            image_width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
            initialize,
        )
    except ImportError:
        raise RuntimeError("CuPy not available for FastBlend")


# Unified factory function
def create_patch_matcher(
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
    backend="auto",
):
    """Create patch matcher with specified backend"""
    if backend == "cuda":
        return create_patch_matcher_cuda(
            height,
            width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            random_search_steps,
            random_search_range,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
        )
    elif backend == "cupy":
        return create_patch_matcher_cupy(
            height,
            width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            random_search_steps,
            random_search_range,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
        )
    elif backend == "auto":
        try:
            return create_patch_matcher_cuda(
                height,
                width,
                channel,
                minimum_patch_size,
                threads_per_block,
                num_iter,
                gpu_id,
                guide_weight,
                random_search_steps,
                random_search_range,
                use_mean_target_style,
                use_pairwise_patch_error,
                tracking_window_size,
            )
        except RuntimeError:
            return create_patch_matcher_cupy(
                height,
                width,
                channel,
                minimum_patch_size,
                threads_per_block,
                num_iter,
                gpu_id,
                guide_weight,
                random_search_steps,
                random_search_range,
                use_mean_target_style,
                use_pairwise_patch_error,
                tracking_window_size,
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_pyramid_patch_matcher(
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
    backend="auto",
):
    """Create pyramid patch matcher with specified backend"""
    if backend == "cuda":
        return create_pyramid_patch_matcher_cuda(
            image_height,
            image_width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
            initialize,
        )
    elif backend == "cupy":
        return create_pyramid_patch_matcher_cupy(
            image_height,
            image_width,
            channel,
            minimum_patch_size,
            threads_per_block,
            num_iter,
            gpu_id,
            guide_weight,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
            initialize,
        )
    elif backend == "auto":
        try:
            return create_pyramid_patch_matcher_cuda(
                image_height,
                image_width,
                channel,
                minimum_patch_size,
                threads_per_block,
                num_iter,
                gpu_id,
                guide_weight,
                use_mean_target_style,
                use_pairwise_patch_error,
                tracking_window_size,
                initialize,
            )
        except RuntimeError:
            return create_pyramid_patch_matcher_cupy(
                image_height,
                image_width,
                channel,
                minimum_patch_size,
                threads_per_block,
                num_iter,
                gpu_id,
                guide_weight,
                use_mean_target_style,
                use_pairwise_patch_error,
                tracking_window_size,
                initialize,
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")


class PatchMatcherCUDA:
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
        if not is_available():
            raise RuntimeError("FastBlend CUDA extension not available")

        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
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

        # Grid and block dimensions for CUDA kernels
        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block,
        )
        self.block = (threads_per_block, threads_per_block)

    def pad_image(self, image):
        """Pad image with reflection padding"""
        # image shape: (B, H, W, C)
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
        """Remove padding from image"""
        pad_size = self.pad_size
        return image[:, pad_size:-pad_size, pad_size:-pad_size, :].contiguous()

    def apply_nnf_to_image(self, nnf, source):
        """Apply NNF to remap source image to target"""
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
        target = remap(
            source.contiguous(), nnf.contiguous(), self.patch_size, self.pad_size
        )
        return target

    def get_patch_error(self, source, nnf, target):
        """Compute patch error between source and target using NNF"""
        batch_size = source.shape[0]
        error = patch_error(
            source.contiguous(),
            nnf.contiguous(),
            target.contiguous(),
            self.patch_size,
            self.pad_size,
        )
        return error

    def get_pairwise_patch_error(self, source, nnf):
        """Compute pairwise patch error for tracking"""
        batch_size = source.shape[0] // 2
        source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
        source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
        error = pairwise_patch_error(
            source_a, nnf_a, source_b, nnf_b, self.patch_size, self.pad_size
        )
        error = error.repeat_interleave(2, dim=0)
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        """Compute combined error (guide + style)"""
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        if self.use_mean_target_style:
            target_style = self.apply_nnf_to_image(nnf, source_style)
            target_style = target_style.mean(dim=0, keepdim=True)
            target_style = target_style.repeat(source_guide.shape[0], 1, 1, 1)
        if self.use_pairwise_patch_error:
            error_style = self.get_pairwise_patch_error(source_style, nnf)
        else:
            error_style = self.get_patch_error(source_style, nnf, target_style)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        """Clamp NNF values to valid bounds"""
        nnf[..., 0] = torch.clamp(nnf[..., 0], 0, self.height - 1)
        nnf[..., 1] = torch.clamp(nnf[..., 1], 0, self.width - 1)
        return nnf

    def random_step(self, nnf, r):
        """Apply random perturbation to NNF"""
        batch_size = nnf.shape[0]
        step = torch.randint(
            -r,
            r + 1,
            (batch_size, self.height, self.width, 2),
            dtype=torch.int32,
            device=nnf.device,
        )
        upd_nnf = self.clamp_bound(nnf + step)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        """Apply neighborhood propagation step"""
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
        upd_nnf = self.clamp_bound(upd_nnf)
        return upd_nnf

    def shift_nnf(self, nnf, d):
        """Shift NNF for tracking"""
        if d > 0:
            d = min(nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[d:], nnf[-1:].repeat(d, 1, 1, 1)], dim=0)
        else:
            d = max(-nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[:1].repeat(-d, 1, 1, 1), nnf[:-d]], dim=0)
        return upd_nnf

    def track_step(self, nnf, d):
        """Apply tracking step"""
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
        """Update NNF if new candidate is better"""
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
        """Run propagation step in all 4 directions"""
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
        """Run random search steps"""
        for i in range(self.random_search_steps):
            upd_nnf = self.random_step(nnf, self.random_search_range)
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
        """Run tracking steps"""
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
        """Run one iteration of patch matching"""
        nnf, err = self.propagation(
            source_guide, target_guide, source_style, target_style, nnf, err
        )
        nnf, err = self.random_search(
            source_guide, target_guide, source_style, target_style, nnf, err
        )
        nnf, err = self.track(
            source_guide, target_guide, source_style, target_style, nnf, err
        )
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        """Estimate NNF for a single level"""
        with torch.cuda.device(self.gpu_id):
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)

            for it in range(self.num_iter):
                self.patch_size = self.patch_size_list[it]
                target_style = self.apply_nnf_to_image(nnf, source_style)
                err = self.get_error(
                    source_guide, target_guide, source_style, target_style, nnf
                )
                nnf, err = self.iteration(
                    source_guide, target_guide, source_style, target_style, nnf, err
                )

            target_style = self.unpad_image(self.apply_nnf_to_image(nnf, source_style))
        return nnf, target_style


class PyramidPatchMatcherCUDA:
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
        if not is_available():
            raise RuntimeError("FastBlend CUDA extension not available")

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

        for level in range(self.pyramid_level):
            height = image_height // (2 ** (self.pyramid_level - 1 - level))
            width = image_width // (2 ** (self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(
                PatchMatcherCUDA(
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
        """Resample images for pyramid level"""
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]

        # Images are on GPU as tensors (B, H, W, C)
        # Convert to (B, C, H, W) for interpolation
        images = images.permute(0, 3, 1, 2)
        images_resample = torch.nn.functional.interpolate(
            images, size=(height, width), mode="area"
        )
        # Convert back to (B, H, W, C)
        images_resample = images_resample.permute(0, 2, 3, 1).contiguous()
        return images_resample

    def initialize_nnf(self, batch_size, height, width):
        """Initialize NNF for coarsest level"""
        device = torch.device(f"cuda:{self.gpu_id}")
        if self.initialize == "random":
            nnf = torch.stack(
                [
                    torch.randint(
                        0,
                        height,
                        (batch_size, height, width),
                        device=device,
                        dtype=torch.int32,
                    ),  # Y (height index)
                    torch.randint(
                        0,
                        width,
                        (batch_size, height, width),
                        device=device,
                        dtype=torch.int32,
                    ),  # X (width index)
                ],
                dim=3,
            )
        elif self.initialize == "identity":
            # Create identity mapping
            # NNF format: [y_coord, x_coord] to match reference implementation
            y_coords = (
                torch.arange(height, device=device, dtype=torch.int32)
                .view(height, 1)
                .repeat(1, width)
            )
            x_coords = (
                torch.arange(width, device=device, dtype=torch.int32)
                .view(1, width)
                .repeat(height, 1)
            )
            nnf = torch.stack([y_coords, x_coords], dim=2)  # (H, W, 2) - [y, x] order
            nnf = nnf.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, H, W, 2)
        else:
            raise NotImplementedError()
        return nnf.contiguous()

    def update_nnf(self, nnf, level):
        """Upscale NNF to next level"""
        # Upscale by factor of 2
        nnf = nnf.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) * 2

        # Add offsets to new pixels
        nnf[:, :, 1::2, 0] += 1  # Add 1 to X for odd columns
        nnf[:, 1::2, :, 1] += 1  # Add 1 to Y for odd rows

        # Check size
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[1] or width != nnf.shape[2]:
            # Resize if dimensions don't match exactly (e.g. odd sizes)
            nnf = nnf.permute(0, 3, 1, 2).float()  # (B, 2, H, W)
            nnf = torch.nn.functional.interpolate(
                nnf, size=(height, width), mode="bilinear", align_corners=False
            )
            nnf = nnf.permute(0, 2, 3, 1).int()

            # Clamp bounds
            nnf[..., 0] = torch.clamp(nnf[..., 0], 0, width - 1)  # X
            nnf[..., 1] = torch.clamp(nnf[..., 1], 0, height - 1)  # Y

        return nnf.contiguous()

    def apply_nnf_to_image(self, nnf, image):
        """Apply NNF to image (used for final output)"""
        with torch.cuda.device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            image = self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
        return image

    def estimate_nnf(self, source_guide, target_guide, source_style):
        """Run pyramid patch matching"""
        with torch.cuda.device(self.gpu_id):
            # Convert to tensors if needed
            if isinstance(source_guide, np.ndarray):
                source_guide = torch.from_numpy(source_guide).float().cuda(self.gpu_id)
            if isinstance(target_guide, np.ndarray):
                target_guide = torch.from_numpy(target_guide).float().cuda(self.gpu_id)
            if isinstance(source_style, np.ndarray):
                source_style = torch.from_numpy(source_style).float().cuda(self.gpu_id)

            nnf = None
            for level in range(self.pyramid_level):
                if level == 0:
                    nnf = self.initialize_nnf(
                        source_guide.shape[0],
                        self.pyramid_heights[0],
                        self.pyramid_widths[0],
                    )
                else:
                    nnf = self.update_nnf(nnf, level)

                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)

                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )

        return nnf, target_style


# Wrapper class to provide PyramidPatchMatcher interface compatible with third-party code
class PyramidPatchMatcher:
    """
    Pyramid patch matcher compatible with third-party interpolation interface.

    This adapts the FastBlend patch matching system to work with interpolation algorithms.
    """

    def __init__(
        self,
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block=8,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity",
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.initialize = initialize
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        # Try to use CUDA implementation first, fallback to cupy
        try:
            if is_available():
                self.patch_matcher = PyramidPatchMatcherCUDA(
                    image_height,
                    image_width,
                    channel,
                    minimum_patch_size,
                    threads_per_block,
                    num_iter,
                    gpu_id,
                    guide_weight,
                    use_mean_target_style,
                    use_pairwise_patch_error,
                    tracking_window_size,
                    initialize,
                )
                self.backend = "cuda"
            else:
                raise ImportError("CUDA extension not available")
        except ImportError:
            try:
                self.patch_matcher = PyramidPatchMatcherCupy(
                    image_height,
                    image_width,
                    channel,
                    minimum_patch_size,
                    threads_per_block,
                    num_iter,
                    gpu_id,
                    guide_weight,
                    use_mean_target_style,
                    use_pairwise_patch_error,
                    tracking_window_size,
                    initialize,
                )
                self.backend = "cupy"
            except ImportError:
                raise ImportError(
                    "Neither CUDA extension nor CuPy available for PyramidPatchMatcher"
                )

    def estimate_nnf(self, source_guide, target_guide, source_style):
        """
        Estimate nearest neighbor field for patch matching.

        Args:
            source_guide: Source guide frames (batch, height, width, channels)
            target_guide: Target guide frames (batch, height, width, channels)
            source_style: Source style frames (batch, height, width, channels)

        Returns:
            Tuple of (nnf, target_style) where target_style contains the interpolated results
        """
        return self.patch_matcher.estimate_nnf(source_guide, target_guide, source_style)
