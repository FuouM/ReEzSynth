// ReEzSynth/ezsynth/ebsynth_extension/cuda_launch.h
//
// Torch-free interface between the PyTorch-facing glue (cuda_dispatch.cpp, compiled
// by the host compiler with PyTorch's C++ standard) and the CUDA code (*.cu,
// compiled by nvcc as C++17 without any PyTorch headers). See accessor.h.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "accessor.h"

namespace ebs
{

// Device buffers for one synthesis level. Every tensor is contiguous and on the
// same device; scratch buffers are allocated (and initialized) by the caller.
struct LevelBuffers
{
    PackedAccessor32<int32_t, 3> nnf;   // (target_h, target_w, 2), updated in place
    PackedAccessor32<float, 2> error;   // (target_h, target_w), output
    PackedAccessor32<uint8_t, 3> output; // (target_h, target_w, C), output
    PackedAccessor32<uint8_t, 3> source_style;
    PackedAccessor32<uint8_t, 3> source_guide;
    PackedAccessor32<uint8_t, 3> target_guide;
    PackedAccessor32<uint8_t, 3> target_modulation; // == source_guide when unused
    bool use_modulation;
    PackedAccessor32<float, 1> style_weights;
    PackedAccessor32<float, 1> guide_weights;

    // Scratch
    PackedAccessor32<int32_t, 2> omega;       // (source_h, source_w), zero-filled
    PackedAccessor32<uint8_t, 3> style_temp;  // like output, zero-filled
    PackedAccessor32<uint8_t, 3> style_prev;  // like output, zero-filled
    PackedAccessor32<uint8_t, 2> mask;        // (target_h, target_w), filled with 255
    PackedAccessor32<uint8_t, 2> mask2;       // (target_h, target_w)
    PackedAccessor64<double, 2> source_sat;    // NCC only: (source_h, source_w)
    PackedAccessor64<double, 2> source_sq_sat;
    PackedAccessor64<double, 2> target_sat;    // NCC only: (target_h, target_w)
    PackedAccessor64<double, 2> target_sq_sat;
    double *sat_scratch; // NCC only: max(source, target) h * w doubles

    void *rand_states; // >= target_h * target_w curandState entries
};

struct LevelParams
{
    float uniformity_weight;
    int patch_size;
    int vote_mode;
    int num_search_vote_iters;
    int num_patch_match_iters;
    int stop_threshold;
    float search_pruning_threshold;
    int cost_function_mode;
    bool use_bilateral;
    float sigma_spatial;
    float sigma_color;
    int n_size_step;
};

// Enqueue one full level (search/vote iterations) on ``stream``. Returns the first
// launch error; execution errors surface at the next synchronizing call.
cudaError_t run_level_cuda(LevelBuffers b, const LevelParams &p, cudaStream_t stream);

// sizeof(curandState); kept out of the host glue so it needs no curand headers.
std::size_t curand_state_size();

// Seed ``num_states`` curandState entries at ``states`` (deterministic, seed 1337).
cudaError_t init_rand_states_cuda_launch(void *states, int64_t num_states, cudaStream_t stream);

} // namespace ebs
