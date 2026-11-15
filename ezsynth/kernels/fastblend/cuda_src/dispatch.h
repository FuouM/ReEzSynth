// FastBlend CUDA dispatch header
#pragma once
#include <torch/extension.h>

// CUDA dispatch functions for FastBlend operations
void fastblend_cuda_remap(
    torch::Tensor target_style,
    torch::Tensor source_style,
    torch::Tensor nnf,
    int patch_size,
    int pad_size
);

void fastblend_cuda_patch_error(
    torch::Tensor error,
    torch::Tensor source,
    torch::Tensor nnf,
    torch::Tensor target,
    int patch_size,
    int pad_size
);

void fastblend_cuda_pairwise_patch_error(
    torch::Tensor error,
    torch::Tensor source_a,
    torch::Tensor nnf_a,
    torch::Tensor source_b,
    torch::Tensor nnf_b,
    int patch_size,
    int pad_size
);