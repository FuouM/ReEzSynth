// ReEzSynth/ezsynth/ebsynth_extension/patchmatch.h
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "accessor.h"

// Forward declarations from other modules
__device__ void update_omega(
    ebs::PackedAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec);

__device__ float patch_omega(
    ebs::PackedAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size);

// ===================================================================
//                        PATCHMATCH KERNELS
// ===================================================================

// Try a candidate patch and update NNF if better
__device__ void try_patch(
    int candidate_sx, int candidate_sy,
    int tx, int ty, int patch_size,
    ebs::PackedAccessor32<int32_t, 3> &nnf,
    ebs::PackedAccessor32<float, 2> &error_map,
    ebs::PackedAccessor32<int32_t, 2> &omega_map,
    ebs::PackedAccessor32<uint8_t, 3> source_style,
    ebs::PackedAccessor32<uint8_t, 3> target_style,
    ebs::PackedAccessor32<uint8_t, 3> source_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const ebs::PackedAccessor32<float, 1> style_weights,
    const ebs::PackedAccessor32<float, 1> guide_weights,
    float uniformity_weight, int cost_function_mode,
    ebs::PackedAccessor64<double, 2> source_style_sat,
    ebs::PackedAccessor64<double, 2> source_style_sq_sat,
    ebs::PackedAccessor64<double, 2> target_style_sat,
    ebs::PackedAccessor64<double, 2> target_style_sq_sat,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);

// Compute initial error for all patches
__global__ void compute_initial_error_kernel(
    ebs::PackedAccessor32<int32_t, 3> nnf,
    ebs::PackedAccessor32<float, 2> error_map,
    ebs::PackedAccessor32<uint8_t, 3> source_style,
    ebs::PackedAccessor32<uint8_t, 3> target_style,
    ebs::PackedAccessor32<uint8_t, 3> source_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int patch_size,
    const ebs::PackedAccessor32<float, 1> style_weights,
    const ebs::PackedAccessor32<float, 1> guide_weights,
    int cost_function_mode,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);

// Propagation step - spatial coherence
__global__ void propagation_step_kernel(
    ebs::PackedAccessor32<int32_t, 3> nnf,
    ebs::PackedAccessor32<float, 2> error_map,
    ebs::PackedAccessor32<int32_t, 2> omega_map,
    ebs::PackedAccessor32<uint8_t, 3> source_style,
    ebs::PackedAccessor32<uint8_t, 3> target_style,
    ebs::PackedAccessor32<uint8_t, 3> source_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const ebs::PackedAccessor32<float, 1> style_weights,
    const ebs::PackedAccessor32<float, 1> guide_weights,
    int patch_size, bool is_odd, float uniformity_weight,
    ebs::PackedAccessor32<uint8_t, 2> mask,
    int cost_function_mode,
    ebs::PackedAccessor64<double, 2> source_style_sat,
    ebs::PackedAccessor64<double, 2> source_style_sq_sat,
    ebs::PackedAccessor64<double, 2> target_style_sat,
    ebs::PackedAccessor64<double, 2> target_style_sq_sat,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);

// Random search step - global exploration
__global__ void random_search_step_kernel(
    ebs::PackedAccessor32<int32_t, 3> nnf,
    ebs::PackedAccessor32<float, 2> error_map,
    ebs::PackedAccessor32<int32_t, 2> omega_map,
    ebs::PackedAccessor32<uint8_t, 3> source_style,
    ebs::PackedAccessor32<uint8_t, 3> target_style,
    ebs::PackedAccessor32<uint8_t, 3> source_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_guide,
    ebs::PackedAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const ebs::PackedAccessor32<float, 1> style_weights,
    const ebs::PackedAccessor32<float, 1> guide_weights,
    int patch_size, int radius, float uniformity_weight, curandState *states,
    ebs::PackedAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold,
    int cost_function_mode,
    ebs::PackedAccessor64<double, 2> source_style_sat,
    ebs::PackedAccessor64<double, 2> source_style_sq_sat,
    ebs::PackedAccessor64<double, 2> target_style_sat,
    ebs::PackedAccessor64<double, 2> target_style_sq_sat,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step);
