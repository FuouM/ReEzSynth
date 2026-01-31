// ReEzSynth/ebsynth_extension/omega_ops.h
#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>

// ===================================================================
//                        UNIFORMITY KERNELS
// ===================================================================

// Update omega map by incrementing/decrementing patch region
__device__ void update_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec);

// Compute omega score for a patch
__device__ float patch_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size);

// Initialize omega map from NNF
__global__ void populate_initial_omega_kernel(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size);
