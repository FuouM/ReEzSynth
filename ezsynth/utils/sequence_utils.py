"""Keyframe ranges and synthesis modes for multi-pass blending."""

from dataclasses import dataclass


@dataclass
class SynthesisSequence:
    MODE_FWD = "forward"
    MODE_REV = "reverse"
    MODE_BLN = "blend"

    start_frame: int
    end_frame: int
    mode: str
    style_indices: list[int]

    def __repr__(self) -> str:
        return f"Sequence(start={self.start_frame}, end={self.end_frame}, mode='{self.mode}', styles={self.style_indices})"


def create_sequences(
    num_frames: int, style_indices: list[int]
) -> list[SynthesisSequence]:
    """Build ``SynthesisSequence`` objects covering the full frame range from style keyframes."""
    keyframes = _valid_keyframes(num_frames, style_indices)
    if keyframes is None:
        return _default_sequence(num_frames)

    sequences = _endpoint_sequences(keyframes)
    for (left_style_idx, left_frame), (right_style_idx, right_frame) in zip(
        keyframes, keyframes[1:]
    ):
        sequences.append(
            SynthesisSequence(
                left_frame,
                right_frame,
                SynthesisSequence.MODE_BLN,
                [left_style_idx, right_style_idx],
            )
        )
    _append_tail_sequence(sequences, num_frames, keyframes)

    print("Defined sequences:")
    for seq in sequences:
        print(f"  - {seq}")

    return sequences


def create_directional_sequences(
    num_frames: int, style_indices: list[int]
) -> list[SynthesisSequence]:
    """Build a fast no-blend schedule: each interval runs from its left anchor."""
    keyframes = _valid_keyframes(num_frames, style_indices)
    if keyframes is None:
        return _default_sequence(num_frames)

    sequences = _endpoint_sequences(keyframes)
    for (left_style_idx, left_frame), (_, right_frame) in zip(keyframes, keyframes[1:]):
        sequences.append(
            SynthesisSequence(
                left_frame,
                right_frame,
                SynthesisSequence.MODE_FWD,
                [left_style_idx],
            )
        )
    _append_tail_sequence(sequences, num_frames, keyframes)

    print("Defined directional sequences:")
    for seq in sequences:
        print(f"  - {seq}")

    return sequences


def _valid_keyframes(
    num_frames: int,
    style_indices: list[int],
) -> list[tuple[int, int]] | None:
    keyframes = list(enumerate(sorted(set(style_indices))))
    if not keyframes or keyframes[0][1] < 0 or keyframes[-1][1] >= num_frames:
        print(
            "Warning: No valid style frames provided or indices out of bounds. Defaulting to a single forward pass."
        )
        return None
    return keyframes


def _default_sequence(num_frames: int) -> list[SynthesisSequence]:
    return [SynthesisSequence(0, num_frames - 1, SynthesisSequence.MODE_FWD, [0])]


def _endpoint_sequences(keyframes: list[tuple[int, int]]) -> list[SynthesisSequence]:
    sequences: list[SynthesisSequence] = []
    _, first_frame = keyframes[0]
    if first_frame > 0:
        sequences.append(
            SynthesisSequence(0, first_frame, SynthesisSequence.MODE_REV, [0])
        )
    return sequences


def _append_tail_sequence(
    sequences: list[SynthesisSequence],
    num_frames: int,
    keyframes: list[tuple[int, int]],
) -> None:
    last_style_idx, last_frame = keyframes[-1]
    if last_frame < num_frames - 1:
        sequences.append(
            SynthesisSequence(
                last_frame,
                num_frames - 1,
                SynthesisSequence.MODE_FWD,
                [last_style_idx],
            )
        )
