"""
Interpolation mode runner for FastBlend.

This module implements keyframe interpolation using patch matching between keyframes.
"""

import numpy as np
from tqdm import tqdm

from .patch_match import PyramidPatchMatcher


class InterpolationModeRunner:
    """Runner for interpolating frames between keyframes using FastBlend."""

    def __init__(self):
        pass

    def get_index_dict(self, index_style):
        """Create mapping from keyframe index to style frame index."""
        index_dict = {}
        for i, index in enumerate(index_style):
            index_dict[index] = i
        return index_dict

    def get_weight(self, l, m, r):
        """Calculate blending weights based on distance from keyframes."""
        weight_l, weight_r = abs(m - r), abs(m - l)
        if weight_l + weight_r == 0:
            weight_l, weight_r = 0.5, 0.5
        else:
            weight_l, weight_r = (
                weight_l / (weight_l + weight_r),
                weight_r / (weight_l + weight_r),
            )
        return weight_l, weight_r

    def get_task_group(self, index_style, n):
        """Group interpolation tasks by segments between keyframes."""
        task_group = []
        index_style = sorted(index_style)

        # Handle frames before first keyframe
        if index_style[0] > 0:
            tasks = []
            for m in range(index_style[0]):
                tasks.append((index_style[0], m, index_style[0]))
            task_group.append(tasks)

        # Handle frames between keyframes
        for l, r in zip(index_style[:-1], index_style[1:]):
            tasks = []
            for m in range(l, r):
                tasks.append((l, m, r))
            task_group.append(tasks)

        # Handle frames after last keyframe
        tasks = []
        for m in range(index_style[-1], n):
            tasks.append((index_style[-1], m, index_style[-1]))
        task_group.append(tasks)

        return task_group

    def run(
        self,
        frames_guide,
        frames_style,
        index_style,
        batch_size,
        ebsynth_config,
        progress_callback=None,
    ):
        """
        Run interpolation between keyframes.

        Args:
            frames_guide: List of guide frames (original content)
            frames_style: List of keyframe style frames (stylized keyframes)
            index_style: Indices of keyframes in frames_guide
            batch_size: Processing batch size
            ebsynth_config: Patch matching configuration
            progress_callback: Optional progress callback

        Returns:
            List of interpolated frames
        """
        # Initialize patch matching engine
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            use_mean_target_style=False,
            use_pairwise_patch_error=True,
            **ebsynth_config,
        )

        # Create task groups
        index_dict = self.get_index_dict(index_style)
        task_group = self.get_task_group(index_style, len(frames_guide))

        # Initialize result frames
        result_frames = [None] * len(frames_guide)

        # Set keyframes directly (ensure they are numpy arrays)
        for kf_idx in index_style:
            style_idx = index_dict[kf_idx]
            keyframe = frames_style[style_idx]
            # Convert to numpy if it's a tensor/array
            if hasattr(keyframe, "cpu"):  # PyTorch tensor
                keyframe = keyframe.cpu().numpy()
            elif hasattr(keyframe, "get"):  # CuPy array
                keyframe = keyframe.get()
            result_frames[kf_idx] = keyframe

        # Calculate total number of frames to process
        total_frames_to_process = len(frames_guide) - len(index_style)

        # Process each task group
        total_frames_processed = 0
        with tqdm(total=total_frames_to_process, desc="Interpolating frames") as pbar:
            for tasks in task_group:
                if not tasks:
                    continue

                index_start, index_end = (
                    min([i[1] for i in tasks]),
                    max([i[1] for i in tasks]),
                )

                # Process tasks in batches
                for batch_id in range(0, len(tasks), batch_size):
                    tasks_batch = tasks[batch_id : min(batch_id + batch_size, len(tasks))]

                    # Prepare batch data for patch matching
                    source_guide, target_guide, source_style = [], [], []

                    for l, m, r in tasks_batch:
                        # Add left keyframe -> target frame matching
                        source_guide.append(frames_guide[l])
                        target_guide.append(frames_guide[m])
                        source_style.append(frames_style[index_dict[l]])

                        # Add right keyframe -> target frame matching
                        source_guide.append(frames_guide[r])
                        target_guide.append(frames_guide[m])
                        source_style.append(frames_style[index_dict[r]])

                    # Convert to numpy arrays
                    source_guide = np.stack(source_guide).astype(np.float32)
                    target_guide = np.stack(target_guide).astype(np.float32)
                    source_style = np.stack(source_style).astype(np.float32)

                    # Run patch matching
                    _, target_style = patch_match_engine.estimate_nnf(
                        source_guide, target_guide, source_style
                    )

                    # Process results - blend left and right interpolations
                    for i, (l, m, r) in enumerate(tasks_batch):
                        # Get the two interpolated results (left and right)
                        frame_l = target_style[i * 2]
                        frame_r = target_style[i * 2 + 1]

                        # Convert to numpy if they are tensors/arrays
                        if hasattr(frame_l, "cpu"):  # PyTorch tensor
                            frame_l = frame_l.cpu().numpy()
                        elif hasattr(frame_l, "get"):  # CuPy array
                            frame_l = frame_l.get()

                        if hasattr(frame_r, "cpu"):  # PyTorch tensor
                            frame_r = frame_r.cpu().numpy()
                        elif hasattr(frame_r, "get"):  # CuPy array
                            frame_r = frame_r.get()

                        # Blend based on distance from keyframes
                        weight_l, weight_r = self.get_weight(l, m, r)
                        blended_frame = frame_l * weight_l + frame_r * weight_r

                        # Clip and convert to uint8
                        blended_frame = np.clip(blended_frame, 0, 255).astype(np.uint8)
                        result_frames[m] = blended_frame

                        total_frames_processed += 1
                        pbar.update(1)

                        # Keep progress callback for backward compatibility
                        if progress_callback:
                            progress_callback(
                                total_frames_processed, total_frames_to_process
                            )

        return result_frames


class InterpolationModeSingleFrameRunner:
    """Runner for single keyframe interpolation (experimental)."""

    def __init__(self):
        pass

    def run(
        self,
        frames_guide,
        frames_style,
        index_style,
        batch_size,
        ebsynth_config,
        progress_callback=None,
    ):
        """Run interpolation with single keyframe."""
        # Check input
        tracking_window_size = ebsynth_config["tracking_window_size"]
        if tracking_window_size * 2 >= batch_size:
            raise ValueError(
                "batch_size should be larger than tracking_window_size * 2"
            )

        frame_style = frames_style[0]
        frame_guide = frames_guide[index_style[0]]

        patch_match_engine = PyramidPatchMatcher(
            image_height=frame_style.shape[0],
            image_width=frame_style.shape[1],
            channel=3,
            **ebsynth_config,
        )

        # Initialize result frames
        result_frames = [None] * len(frames_guide)
        # Set the keyframe (ensure it's a numpy array)
        keyframe = frame_style
        if hasattr(keyframe, "cpu"):  # PyTorch tensor
            keyframe = keyframe.cpu().numpy()
        elif hasattr(keyframe, "get"):  # CuPy array
            keyframe = keyframe.get()
        result_frames[index_style[0]] = keyframe

        # Process frames in batches
        frame_id = 0
        n = len(frames_guide)
        frames_to_process = n - 1  # Subtract 1 for the keyframe

        with tqdm(total=frames_to_process, desc="Processing single keyframe interpolation") as pbar:
            for i in range(0, n, batch_size - tracking_window_size * 2):
                if i + batch_size > n:
                    l, r = max(n - batch_size, 0), n
                else:
                    l, r = i, i + batch_size

                source_guide = np.stack([frame_guide] * (r - l)).astype(np.float32)
                target_guide = np.stack([frames_guide[j] for j in range(l, r)]).astype(
                    np.float32
                )
                source_style = np.stack([frame_style] * (r - l)).astype(np.float32)

                _, target_style = patch_match_engine.estimate_nnf(
                    source_guide, target_guide, source_style
                )

                for j, frame_idx in enumerate(range(l, r)):
                    if frame_idx == index_style[0]:  # Skip the keyframe itself
                        continue

                    frame = target_style[j]
                    # Convert to numpy if it's a tensor/array
                    if hasattr(frame, "cpu"):  # PyTorch tensor
                        frame = frame.cpu().numpy()
                    elif hasattr(frame, "get"):  # CuPy array
                        frame = frame.get()

                    result_frames[frame_idx] = np.clip(frame, 0, 255).astype(np.uint8)

                    pbar.update(1)

                    # Keep progress callback for backward compatibility
                    if progress_callback:
                        progress_callback(frame_idx + 1, n)

        return result_frames
