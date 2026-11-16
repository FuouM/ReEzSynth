"""
Interpolation mode runner for FastBlend.

This module implements keyframe interpolation using patch matching between keyframes.
"""

import itertools

import numpy as np
from tqdm import tqdm

from .patch_match import PyramidPatchMatcher


class InterpolationModeRunner:
    """Runner for interpolating frames between keyframes using FastBlend."""

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
        for l, r in itertools.pairwise(index_style):
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
        patch_matcher_config,
        progress_callback=None,
    ):
        """
        Run interpolation between keyframes.

        Args:
            frames_guide: List of guide frames (original content)
            frames_style: List of keyframe style frames (stylized keyframes)
            index_style: Indices of keyframes in frames_guide
            batch_size: Processing batch size
            patch_matcher_config: Settings for :class:`~FastBlend.src.patch_match.PyramidPatchMatcher`
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
            **patch_matcher_config,
        )

        # Keyframe index -> position in frames_style; keyframes are copied as-is.
        index_dict = {index: i for i, index in enumerate(index_style)}
        task_group = self.get_task_group(index_style, len(frames_guide))

        result_frames = [None] * len(frames_guide)
        for kf_idx in index_style:
            result_frames[kf_idx] = frames_style[index_dict[kf_idx]]

        # Calculate total number of frames to process
        total_frames_to_process = len(frames_guide) - len(index_style)

        # Process each task group
        total_frames_processed = 0
        with tqdm(total=total_frames_to_process, desc="Interpolating frames") as pbar:
            for tasks in task_group:
                if not tasks:
                    continue

                # Process tasks in batches
                for batch_id in range(0, len(tasks), batch_size):
                    tasks_batch = tasks[
                        batch_id : min(batch_id + batch_size, len(tasks))
                    ]

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
                    target_style = target_style.cpu().numpy()

                    # Blend the left and right matches by distance from the keyframes.
                    for i, (l, m, r) in enumerate(tasks_batch):
                        weight_l, weight_r = self.get_weight(l, m, r)
                        blended_frame = (
                            target_style[i * 2] * weight_l
                            + target_style[i * 2 + 1] * weight_r
                        )
                        result_frames[m] = np.clip(blended_frame, 0, 255).astype(
                            np.uint8
                        )

                        total_frames_processed += 1
                        pbar.update(1)

                        if progress_callback:
                            progress_callback(
                                total_frames_processed, total_frames_to_process
                            )

        return result_frames
