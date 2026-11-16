import logging
from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm

from ezsynth.kernels.fastblend import build_pyramid_matcher, load, resolve

logger = logging.getLogger(__name__)


class GPUAccumulator:
    def __init__(self, device):
        self.device = device
        self.buffer: dict[int, tuple[torch.Tensor, float]] = {}

    def update(self, target_idx: int, result_tensor: torch.Tensor):
        """Add an (H, W, C) float32 result for ``target_idx`` (sum of results and count).

        ``get_final`` adds the base style frame with weight 1, so the result is the
        running average ``(base + sum(results)) / (1 + count)``.
        """
        if target_idx not in self.buffer:
            self.buffer[target_idx] = (result_tensor.clone(), 1.0)
        else:
            acc_frame, acc_weight = self.buffer[target_idx]
            acc_frame.add_(result_tensor)
            self.buffer[target_idx] = (acc_frame, acc_weight + 1.0)

    def get_final(self, target_idx: int, base_frame_cpu: np.ndarray) -> np.ndarray:
        """Pop ``target_idx`` and return ``(base + sum) / (1 + count)`` as float32 numpy."""
        acc_frame, acc_weight = self.buffer.pop(target_idx)
        base_frame = (
            torch.from_numpy(base_frame_cpu)
            .to(self.device, dtype=torch.float32)
            .contiguous()
        )
        return ((base_frame + acc_frame) / (1.0 + acc_weight)).cpu().numpy()


class FrameCache:
    def __init__(self, frames: list[np.ndarray], device, max_size_mb=2000):
        self.frames = frames
        self.device = device
        self.cache: dict[int, torch.Tensor] = {}
        self.lru: list[int] = []
        self.max_size_mb = max_size_mb

        # Estimate size of one frame in MB
        if len(frames) > 0:
            h, w, c = frames[0].shape
            self.frame_size_mb = (h * w * c * 4) / (1024 * 1024)  # float32
        else:
            self.frame_size_mb = 1

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """
        Get a batch of frames as a single GPU tensor (B, H, W, C).
        """
        # Identify missing frames
        missing = [idx for idx in indices if idx not in self.cache]

        # Ensure space
        needed_mb = len(missing) * self.frame_size_mb
        current_mb = len(self.cache) * self.frame_size_mb

        while current_mb + needed_mb > self.max_size_mb and self.lru:
            # Evict LRU
            evict_idx = self.lru.pop(0)
            if evict_idx in self.cache:
                del self.cache[evict_idx]
                current_mb -= self.frame_size_mb

        # Load missing
        for idx in missing:
            tensor = (
                torch.from_numpy(self.frames[idx])
                .to(self.device, dtype=torch.float32)
                .contiguous()
            )
            self.cache[idx] = tensor

        # Update LRU
        for idx in indices:
            if idx in self.lru:
                self.lru.remove(idx)
            self.lru.append(idx)

        # Stack
        # Note: torch.stack of GPU tensors is fast
        tensors = [self.cache[idx] for idx in indices]
        return torch.stack(tensors).contiguous()


class BalancedModeRunner:
    def run(
        self,
        frames_guide,
        frames_style,
        batch_size,
        window_size,
        patch_matcher_config,
        desc="Balanced Mode",
        progress_callback: Callable[[int, int], None] | None = None,
        backend="auto",
    ):
        """Blend every frame with the frames within ``window_size`` matched onto it."""
        resolved = resolve(backend)

        suffix = " (auto)" if backend == "auto" else ""
        logger.info(f"FastBlend: Using {resolved} backend{suffix}")

        pm_cfg = {k: v for k, v in patch_matcher_config.items() if k != "backend"}
        patch_match_engine = build_pyramid_matcher(
            resolved,
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            **pm_cfg,
        )
        device = load(resolved).staging_device(
            int(patch_matcher_config.get("gpu_id", 0))
        )

        # (source, target) pairs in target order, so targets complete in order.
        n = len(frames_style)
        tasks = []
        for target in range(n):
            for source in range(target - window_size, target + window_size + 1):
                if source >= 0 and source < n and source != target:
                    tasks.append((source, target))

        total_tasks = len(tasks)
        logger.info(
            f"FastBlend: Processing {total_tasks} frame pairs in {len(frames_style)} frames"
        )

        output = [None for _ in range(n)]
        completed_frames = 0

        accumulator = GPUAccumulator(device)
        # 2 GB LRU cache: ~600 frames at 512x512, ~80 at 1080p (float32).
        guide_cache = FrameCache(frames_guide, device)
        style_cache = FrameCache(frames_style, device)

        with tqdm(total=total_tasks, desc=desc, unit="pairs") as pbar:
            for batch_id in range(0, total_tasks, batch_size):
                tasks_batch = tasks[batch_id : min(batch_id + batch_size, len(tasks))]

                sources = [s for s, t in tasks_batch]
                targets = [t for s, t in tasks_batch]

                s_guide_t = guide_cache.get_batch(sources)
                t_guide_t = guide_cache.get_batch(targets)
                s_style_t = style_cache.get_batch(sources)

                _, target_style_t = patch_match_engine.estimate_nnf(
                    s_guide_t, t_guide_t, s_style_t
                )

                for i, (_source, target) in enumerate(tasks_batch):
                    accumulator.update(target, target_style_t[i])

                    # Window frames of ``target``, itself (the base frame) included.
                    expected_weight = min(n, target + window_size + 1) - max(
                        0, target - window_size
                    )
                    _, current_weight = accumulator.buffer[target]
                    if current_weight + 1 >= expected_weight:
                        output[target] = accumulator.get_final(
                            target, frames_style[target]
                        )
                        completed_frames += 1

                        if progress_callback:
                            progress_callback(completed_frames, n)

                pbar.update(len(tasks_batch))

        # A single-frame sequence has no pairs: its frame is returned unchanged.
        for i in range(n):
            if output[i] is None:
                output[i] = frames_style[i]

        logger.info(f"FastBlend: Completed processing {completed_frames}/{n} frames")
        return output
