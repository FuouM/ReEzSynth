# ReEzSynth/setup.py
from setuptools import setup

# CUDA extension is disabled - using JIT compilation instead
# The extension will be compiled at runtime via ebsynth_torch_loader.py
# This avoids build-time dependencies on CUDA toolkit and C++ compiler

setup(
    name="ebsynth_torch",
    # No ext_modules - JIT compilation is used instead
    # Use: python ebsynth_torch_loader.py to trigger JIT compilation at runtime
)
