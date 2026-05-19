"""ComfyUI custom node entry for ReEzSynth."""

import os
import sys

_NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _NODE_ROOT not in sys.path:
    sys.path.insert(0, _NODE_ROOT)

from .comfy import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
