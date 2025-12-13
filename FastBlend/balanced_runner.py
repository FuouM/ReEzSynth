from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Lazy imports handled inside the class or function to avoid top-level dependency
try:
    from .patch_match import create_pyramid_patch_matcher
except ImportError:
    pass


class GPUAccumulator:
    def __init__(self, device):
        self.device = device
        self.buffer: Dict[int, Tuple[torch.Tensor, float]] = {}

    def update(self, target_idx: int, result_tensor, weight_inc: float = 1.0):
        """
        Accumulate a result into the buffer for target_idx.
        result_tensor: (H, W, C) float32 tensor on GPU or numpy array
        """
        # Convert to tensor if needed
        if isinstance(result_tensor, np.ndarray):
            result_tensor = (
                torch.from_numpy(result_tensor)
                .to(self.device, dtype=torch.float32)
                .contiguous()
            )
        elif not isinstance(result_tensor, torch.Tensor):
            raise TypeError(
                f"result_tensor must be numpy array or torch tensor, got {type(result_tensor)}"
            )

        if target_idx not in self.buffer:
            # First contribution: just store it
            # We store (accumulated_frame, accumulated_weight)
            # To match the lerp logic: frame = frame * (w/(w+1)) + result * (1/(w+1))
            # This is equivalent to weighted average.
            # We can just sum them up and divide by total weight at the end?
            # The original code did iterative update:
            # frame = frame * (weight / (weight + 1)) + result / (weight + 1)
            # This is exactly running average: NewAvg = (OldAvg * OldWeight + NewVal * 1) / (OldWeight + 1)
            # So yes, we can just sum values and sum weights.
            # But wait, original code initialized with `frames_style[target]` if None.
            # "if frame is None: frame = frames_style[target]"
            # So the base is the original style frame with weight 1.
            # Then we add results.

            # We need to handle the initialization logic outside or here.
            # Let's assume the caller handles initialization if needed,
            # or we initialize with 0 and add the base frame separately?
            # The runner logic:
            # frames = [(None, 1) for i in range(n)]
            # ...
            # if frame is None: frame = frames_style[target]
            # frame = frame * ...

            # So effectively: Final = (Original * 1 + Sum(Results)) / (1 + Count)
            # We can store Sum(Results) and Count.

            self.buffer[target_idx] = (result_tensor.clone(), weight_inc)
        else:
            acc_frame, acc_weight = self.buffer[target_idx]
            acc_frame.add_(result_tensor)
            self.buffer[target_idx] = (acc_frame, acc_weight + weight_inc)

    def get_final(self, target_idx: int, base_frame_cpu: np.ndarray) -> np.ndarray:
        """
        Compute final result for target_idx, combining with base frame.
        Returns uint8 numpy array (or float32 if that was input).
        """
        if target_idx not in self.buffer:
            return base_frame_cpu

        acc_frame, acc_weight = self.buffer[target_idx]

        # Move base frame to GPU
        base_frame = (
            torch.from_numpy(base_frame_cpu)
            .to(self.device, dtype=torch.float32)
            .contiguous()
        )

        # Final = (Base * 1 + Sum) / (1 + Weight)
        final_frame = (base_frame + acc_frame) / (1.0 + acc_weight)

        # Clean up
        del self.buffer[target_idx]

        return final_frame.cpu().numpy()

    def clear(self):
        self.buffer.clear()


class FrameCache:
    def __init__(self, frames: List[np.ndarray], device, max_size_mb=2000):
        self.frames = frames
        self.device = device
        self.cache: Dict[int, torch.Tensor] = {}
        self.lru: List[int] = []
        self.max_size_mb = max_size_mb

        # Estimate size of one frame in MB
        if len(frames) > 0:
            h, w, c = frames[0].shape
            self.frame_size_mb = (h * w * c * 4) / (1024 * 1024)  # float32
        else:
            self.frame_size_mb = 1

    def get_batch(self, indices: List[int]) -> torch.Tensor:
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
    def __init__(self):
        pass

    def run(
        self,
        frames_guide,
        frames_style,
        batch_size,
        window_size,
        ebsynth_config,
        desc="Balanced Mode",
        save_path=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        backend="auto",
    ):
        """
        Run FastBlend processing with progress tracking.
        """
        # Check backend availability
        cuda_available = False
        cupy_available = False

        try:
            from .fastblend_extension import is_available as cuda_is_available

            cuda_available = cuda_is_available()
        except ImportError:
            cuda_available = False

        try:
            import cupy

            cupy_available = True
        except ImportError:
            cupy_available = False

        # Select backend based on preference
        if backend == "cuda" and cuda_available:
            print("FastBlend: Using CUDA backend")
        elif backend == "cupy" and cupy_available:
            print("FastBlend: Using CuPy backend")
        elif backend == "auto":
            if cuda_available:
                print("FastBlend: Using CUDA backend (auto-selected)")
            elif cupy_available:
                print("FastBlend: Using CuPy backend (fallback)")
            else:
                raise ImportError("No FastBlend backend available")
        else:
            available_backends = []
            if cuda_available:
                available_backends.append("cuda")
            if cupy_available:
                available_backends.append("cupy")
            raise ImportError(
                f"Requested backend '{backend}' not available. Available: {available_backends}"
            )

        # Setup Engine
        patch_match_engine = create_pyramid_patch_matcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            backend=backend,
            **ebsynth_config,
        )
        device = torch.device("cuda", ebsynth_config.get("gpu_id", 0))

        # Generate all tasks
        n = len(frames_style)
        tasks = []
        for target in range(n):
            for source in range(target - window_size, target + window_size + 1):
                if source >= 0 and source < n and source != target:
                    tasks.append((source, target))

        total_tasks = len(tasks)
        print(
            f"FastBlend: Processing {total_tasks} frame pairs in {len(frames_style)} frames"
        )

        # Sort tasks by target to optimize accumulation locality
        # They are already generated in target order, so that's good.

        output = [None for _ in range(n)]
        completed_frames = 0

        # Buffers
        accumulator = GPUAccumulator(device)
        # Cache size: 2GB default.
        # 512x512x3 float32 is ~3MB. 2GB holds ~600 frames.
        # 1080p is ~24MB. 2GB holds ~80 frames.
        guide_cache = FrameCache(frames_guide, device)
        style_cache = FrameCache(frames_style, device)

        # Process tasks
        with tqdm(total=total_tasks, desc=desc, unit="pairs") as pbar:
            for batch_id in range(0, total_tasks, batch_size):
                tasks_batch = tasks[batch_id : min(batch_id + batch_size, len(tasks))]

                sources = [s for s, t in tasks_batch]
                targets = [t for s, t in tasks_batch]

                # Optimized CUDA Path

                # Fetch data using cache (returns GPU tensors)
                s_guide_t = guide_cache.get_batch(sources)
                t_guide_t = guide_cache.get_batch(targets)
                s_style_t = style_cache.get_batch(sources)

                # Run patch match
                # Note: estimate_nnf expects (B, H, W, C)
                _, target_style_t = patch_match_engine.estimate_nnf(
                    s_guide_t, t_guide_t, s_style_t
                )

                # Accumulate
                for i, (source, target) in enumerate(tasks_batch):
                    result = target_style_t[i]
                    accumulator.update(target, result)

                    # Check completion
                    expected_weight = min(n, target + window_size + 1) - max(
                        0, target - window_size
                    )
                    # We need to track current weight.
                    # GPUAccumulator stores it.
                    _, current_weight = accumulator.buffer[target]

                    # Note: accumulator stores SUM of weights of added frames.
                    # Base frame has weight 1.
                    # So total weight = current_weight + 1.

                    if current_weight + 1 >= expected_weight:
                        # Finalize
                        final_frame = accumulator.get_final(
                            target, frames_style[target]
                        )
                        output[target] = final_frame
                        completed_frames += 1

                        if progress_callback:
                            progress_callback(completed_frames, n)

                # Periodic cache cleanup if needed (FrameCache handles LRU)

                pbar.update(len(tasks_batch))

        # Fill any remaining frames (should be none if logic is correct)
        for i in range(n):
            if output[i] is None:
                # Fallback if something went wrong or single frame
                output[i] = frames_style[i]

        print(f"FastBlend: Completed processing {completed_frames}/{n} frames")
        return output
