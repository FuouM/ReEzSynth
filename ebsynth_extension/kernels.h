// ReEzSynth/ebsynth_extension/kernels.h
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/all.h>

// ===================================================================
//                        HELPER DEFINITIONS
// ===================================================================

#define EBSYNTH_VOTEMODE_PLAIN 0x0001
#define EBSYNTH_VOTEMODE_WEIGHTED 0x0002

// Helper for CUDA vector types
template <int N, typename T>
struct Vec {
  T v[N];
};

// ===================================================================
//                        UNIFORMITY KERNELS
// ===================================================================

__device__ void update_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec);

__device__ float patch_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size);

__global__ void populate_initial_omega_kernel(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size);

// ===================================================================
//                        MASKING KERNELS
// ===================================================================

__global__ void eval_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold);

__global__ void dilate_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size);

// ===================================================================
//                        PATCHMATCH KERNELS
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
    float ebest);

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
    float uniformity_weight);

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
    const torch::PackedTensorAccessor32<float, 1> guide_weights);

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
    torch::PackedTensorAccessor32<uint8_t, 2> mask);

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
    float search_pruning_threshold);

// ===================================================================
//                        VOTING KERNELS
// ===================================================================

__global__ void krnlVotePlain(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size);

__global__ void krnlVoteWeighted(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    int patch_size);


// ===================================================================
//                RNG STATE INITIALIZER KERNEL
// ===================================================================
__global__ void init_rand_states_kernel(curandState* states, int num_states);