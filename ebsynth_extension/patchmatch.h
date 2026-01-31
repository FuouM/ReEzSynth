// ReEzSynth/ebsynth_extension/patchmatch.h
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/all.h>

// Forward declarations from other modules
__device__ void update_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec);

__device__ float patch_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size);

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
    float ebest);

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
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat);

// ===================================================================
//                        PATCHMATCH KERNELS
// ===================================================================

// Try a candidate patch and update NNF if better
__device__ void try_patch(
    int candidate_sx, int candidate_sy,
    int tx, int ty, int patch_size,
    torch::PackedTensorAccessor32<int32_t, 3>& nnf,
    torch::PackedTensorAccessor32<float, 2>& error_map,
    torch::PackedTensorAccessor32<int32_t, 2>& omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float uniformity_weight, int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat);

// Compute initial error for all patches
__global__ void compute_initial_error_kernel(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int cost_function_mode);

// Propagation step - spatial coherence
__global__ void propagation_step_kernel(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, bool is_odd, float uniformity_weight,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat);

// Random search step - global exploration
__global__ void random_search_step_kernel(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, int radius, float uniformity_weight, curandState* states,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat);
