from .nodes_image_synth import (
    ReEzSynthGuide,
    ReEzSynthGuidesAppend,
    ReEzSynthImageSynth,
    ReEzSynthImageSynthConfig,
)

NODE_CLASS_MAPPINGS = {
    "ReEzSynthGuide": ReEzSynthGuide,
    "ReEzSynthGuidesAppend": ReEzSynthGuidesAppend,
    "ReEzSynthImageSynthConfig": ReEzSynthImageSynthConfig,
    "ReEzSynthImageSynth": ReEzSynthImageSynth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReEzSynthGuide": "ReEzSynth Guide",
    "ReEzSynthGuidesAppend": "ReEzSynth Guides Append",
    "ReEzSynthImageSynthConfig": "ReEzSynth Image Synth Config",
    "ReEzSynthImageSynth": "ReEzSynth Image Synth",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
