"""ComfyUI custom node entry for ReEzSynth."""

import os
import sys

_NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _NODE_ROOT not in sys.path:
    sys.path.insert(0, _NODE_ROOT)

# Absolute import: this file is imported both as a ComfyUI custom-node package and,
# during pytest collection, as a bare module without a parent package.
from reezsynth_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
