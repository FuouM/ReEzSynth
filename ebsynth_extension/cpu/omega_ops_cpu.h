// ReEzSynth/ebsynth_extension/cpu/omega_ops_cpu.h
#pragma once

#include <torch/extension.h>
#include <cstdint>

// ===================================================================
//                        UNIFORMITY OPERATIONS
// ===================================================================

// Update omega map by incrementing/decrementing patch region
void update_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec);

// Compute omega score for a patch
float patch_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size);

// Initialize omega map from NNF
void populate_initial_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size,
    int target_h, int target_w);
