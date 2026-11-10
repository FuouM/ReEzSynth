// FastBlend CUDA Kernels Header
//
// Torch-free on purpose: kernels.cu is compiled by nvcc as C++17 without PyTorch
// headers (PyTorch >= 2.12 headers need C++20, which older nvcc cannot compile).
// The PyTorch-facing code lives in dispatch.cpp.

#pragma once

#include <cuda_runtime.h>

// Kernel launchers. Each enqueues on ``stream`` and returns the launch error.
cudaError_t launch_remap_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_style, const int* nnf, float* target_style,
    const int batch_size, cudaStream_t stream
);

cudaError_t launch_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source, const int* nnf, const float* target, float* error,
    const int batch_size, cudaStream_t stream
);

cudaError_t launch_pairwise_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_a, const int* nnf_a, const float* source_b, const int* nnf_b, float* error,
    const int batch_size, cudaStream_t stream
);
