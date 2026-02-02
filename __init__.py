# ReEzSynth Custom Node
"""
ReEzSynth ComfyUI Custom Nodes

This module provides ComfyUI custom nodes for video/image synthesis.
"""

import os
import sys

# Get the absolute path to the ezsynth directory (sibling of this directory)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ezsynth_parent = _current_dir  # ezsynth is in the same directory as this file
if _ezsynth_parent not in sys.path:
    sys.path.insert(0, _ezsynth_parent)

from .comfyui_ezsynth import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
