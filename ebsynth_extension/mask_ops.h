// ReEzSynth/ebsynth_extension/mask_ops.h
#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>

// ===================================================================
//                        MASKING KERNELS
// ===================================================================

// Evaluate mask based on pixel differences between two images
__global__ void eval_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold);

// Dilate mask by patch_size radius
__global__ void dilate_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size);
