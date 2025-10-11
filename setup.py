# ReEzSynth/setup.py
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "-std=c++17"],
}

if sys.platform == "win32":
    # MSVC specific flags
    extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/wd4244", "/wd4267"]
    extra_compile_args["nvcc"] = ["-O3", "--use-local-env", "-std=c++17"]

setup(
    name="ebsynth_torch",
    ext_modules=[
        CUDAExtension(
            name="ebsynth_torch",
            sources=[
                "ebsynth_extension/ext.cpp",
                "ebsynth_extension/dispatch.cu",
                "ebsynth_extension/kernels.cu",
                "ebsynth_extension/integral_image.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
