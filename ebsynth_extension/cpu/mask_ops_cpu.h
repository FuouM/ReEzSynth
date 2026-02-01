// ReEzSynth/ebsynth_extension/cpu/mask_ops_cpu.h
#pragma once

#include <torch/extension.h>
#include <cstdint>

// ===================================================================
//                        MASKING OPERATIONS
// ===================================================================

// Evaluate mask based on pixel differences between two images
void eval_mask_cpu(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold,
    int target_h, int target_w);

// Dilate mask by patch_size radius
void dilate_mask_cpu(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size,
    int target_h, int target_w);
