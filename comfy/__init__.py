from .nodes_image_synth import (
    ReEzSynthGuide,
    ReEzSynthGuidesAppend,
    ReEzSynthImageSynth,
    ReEzSynthImageSynthConfig,
)
from .nodes_video_synth import (
    ReEzSynthVideoStyleKeyframe,
    ReEzSynthVideoStyleKeyframes,
    ReEzSynthVideoStyleKeyframesAppend,
    ReEzSynthVideoSynth,
    ReEzSynthVideoSynthConfig,
)

NODE_CLASS_MAPPINGS = {
    "ReEzSynthGuide": ReEzSynthGuide,
    "ReEzSynthGuidesAppend": ReEzSynthGuidesAppend,
    "ReEzSynthImageSynthConfig": ReEzSynthImageSynthConfig,
    "ReEzSynthImageSynth": ReEzSynthImageSynth,
    "ReEzSynthVideoStyleKeyframe": ReEzSynthVideoStyleKeyframe,
    "ReEzSynthVideoStyleKeyframes": ReEzSynthVideoStyleKeyframes,
    "ReEzSynthVideoStyleKeyframesAppend": ReEzSynthVideoStyleKeyframesAppend,
    "ReEzSynthVideoSynthConfig": ReEzSynthVideoSynthConfig,
    "ReEzSynthVideoSynth": ReEzSynthVideoSynth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReEzSynthGuide": "ReEzSynth Guide",
    "ReEzSynthGuidesAppend": "ReEzSynth Guides Append",
    "ReEzSynthImageSynthConfig": "ReEzSynth Image Synth Config",
    "ReEzSynthImageSynth": "ReEzSynth Image Synth",
    "ReEzSynthVideoStyleKeyframe": "ReEzSynth Video Style Keyframe",
    "ReEzSynthVideoStyleKeyframes": "ReEzSynth Video Style Keyframes",
    "ReEzSynthVideoStyleKeyframesAppend": "ReEzSynth Video Style Keyframes Append",
    "ReEzSynthVideoSynthConfig": "ReEzSynth Video Synth Config",
    "ReEzSynthVideoSynth": "ReEzSynth Video Synth",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
