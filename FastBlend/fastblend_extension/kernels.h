// FastBlend CUDA Kernels Header

#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>

#include <cuda_runtime.h>

// Kernel launchers
void launch_remap_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_style, const int* nnf, float* target_style,
    const int batch_size
);

void launch_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source, const int* nnf, const float* target, float* error,
    const int batch_size
);

void launch_pairwise_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_a, const int* nnf_a, const float* source_b, const int* nnf_b, float* error,
    const int batch_size
);