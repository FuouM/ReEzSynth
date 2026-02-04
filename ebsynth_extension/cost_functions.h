// ReEzSynth/ebsynth_extension/cost_functions.h
#pragma once

#include <cuda_runtime.h>
#include <torch/all.h>

// Cost function mode constants
#define COST_FUNCTION_SSD 0
#define COST_FUNCTION_NCC 1

// ===================================================================
//                        SSD COST FUNCTION
// ===================================================================

__device__ float compute_patch_ssd_split(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);

// ===================================================================
//                        NCC COST FUNCTIONS
// ===================================================================

// NCC using Summed-Area Tables (SATs) for O(1) style stats
__device__ float compute_patch_ncc_sat(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);

// NCC computed directly (O(P^2) for all operations)
__device__ float compute_patch_ncc_split(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest);
