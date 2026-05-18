"""Parse frame indices from filenames; align keyframe files to ordered frame lists."""

import re
from typing import List, Optional, Tuple


def extract_frame_number_from_filename(filename: str) -> Optional[int]:
    """Extract a frame index from ``filename`` using common naming patterns."""
    patterns = [
        r"(\d{5,})\.png$",
        r"(\d{5,})\.jpg$",
        r"(\d{5,})\.jpeg$",
        r"frame[_]?(\d{5,})\.png$",
        r"frame[_]?(\d{5,})\.jpg$",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"(\d{5,})", filename)
    if match:
        return int(match.group(1))

    return None


def match_keyframes_to_frames(
    frame_files: List[str], keyframe_files: List[str]
) -> Tuple[List[Optional[str]], List[int]]:
    """
    Match keyframe filenames to ordered frame filenames by parsed frame number.

    Returns ``(matched_keyframes, keyframe_indices)`` where
    ``matched_keyframes[i]`` is the keyframe basename for frame ``i``, or
    ``None`` if no keyframe shares that index.
    """
    frame_numbers = [extract_frame_number_from_filename(f) for f in frame_files]
    keyframe_numbers = [extract_frame_number_from_filename(f) for f in keyframe_files]

    keyframe_map = {
        num: filename
        for num, filename in zip(keyframe_numbers, keyframe_files)
        if num is not None
    }

    matched_keyframes: List[Optional[str]] = []
    keyframe_indices: List[int] = []

    for i, frame_num in enumerate(frame_numbers):
        if frame_num in keyframe_map:
            matched_keyframes.append(keyframe_map[frame_num])
            keyframe_indices.append(i)
        else:
            matched_keyframes.append(None)

    return matched_keyframes, keyframe_indices
