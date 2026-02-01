// ReEzSynth/ebsynth_extension/cpu/dispatch_cpu.cpp
#include "dispatch_cpu.h"

#include "cost_functions_cpu.h"
#include "omega_ops_cpu.h"
#include "voting_cpu.h"
#include "mask_ops_cpu.h"
#include "patchmatch_cpu.h"
#include "integral_image_cpu.h"

#include <random>
#include <algorithm>

// Vote mode constants
#define EBSYNTH_VOTEMODE_PLAIN 0x0001
#define EBSYNTH_VOTEMODE_WEIGHTED 0x0002

// Cost function mode constants
#define COST_FUNCTION_SSD 0
#define COST_FUNCTION_NCC 1

// ===================================================================
//                MAIN CPU DISPATCH FUNCTION (SINGLE LEVEL)
// ===================================================================
void ebsynth_cpu_run_level(
    torch::Tensor output_image,
    torch::Tensor output_error,
    torch::Tensor nnf,
    torch::Tensor style_level,
    torch::Tensor source_guide_level,
    torch::Tensor target_guide_level,
    torch::Tensor target_modulation_level,
    torch::Tensor style_weights,
    torch::Tensor guide_weights,
    float uniformity_weight,
    int patch_size,
    int vote_mode,
    int num_search_vote_iters,
    int num_patch_match_iters,
    int stop_threshold,
    torch::Tensor rand_states_tensor,
    float search_pruning_threshold,
    int cost_function_mode) {

    const int source_h = style_level.size(0);
    const int source_w = style_level.size(1);
    const int target_h = target_guide_level.size(0);
    const int target_w = target_guide_level.size(1);

    auto nnf_acc = nnf.packed_accessor32<int32_t, 3>();
    auto error_acc = output_error.packed_accessor32<float, 2>();
    auto source_style_acc = style_level.packed_accessor32<uint8_t, 3>();
    auto source_guide_acc = source_guide_level.packed_accessor32<uint8_t, 3>();
    auto target_guide_acc = target_guide_level.packed_accessor32<uint8_t, 3>();

    bool use_modulation = target_modulation_level.numel() > 0;
    auto target_modulation_guide_acc = use_modulation ?
        target_modulation_level.packed_accessor32<uint8_t, 3>() : source_guide_acc;

    auto style_weights_acc = style_weights.packed_accessor32<float, 1>();
    auto guide_weights_acc = guide_weights.packed_accessor32<float, 1>();

    // Initialize omega map
    torch::Tensor omega_map = torch::zeros({source_h, source_w}, torch::kInt32);
    auto omega_acc = omega_map.packed_accessor32<int32_t, 2>();
    populate_initial_omega_cpu(omega_acc, nnf_acc, patch_size, target_h, target_w);

    // Temporary tensors
    torch::Tensor target_style_temp = torch::zeros_like(output_image);
    torch::Tensor target_style_prev = torch::zeros_like(output_image);
    torch::Tensor mask = torch::full({target_h, target_w}, 255, torch::kUInt8);
    torch::Tensor mask2 = torch::empty_like(mask);

    auto target_style_temp_acc = target_style_temp.packed_accessor32<uint8_t, 3>();
    auto target_style_prev_acc = target_style_prev.packed_accessor32<uint8_t, 3>();
    auto mask_acc = mask.packed_accessor32<uint8_t, 2>();
    auto mask2_acc = mask2.packed_accessor32<uint8_t, 2>();

    // Precomputation for NCC (SATs)
    auto sat_options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor source_style_sat = torch::empty({source_h, source_w}, sat_options);
    torch::Tensor source_style_sq_sat = torch::empty({source_h, source_w}, sat_options);

    if (cost_function_mode == COST_FUNCTION_NCC) {
        compute_integral_image_cpu(source_style_sat, style_level, PREP_GRAY);
        compute_integral_image_cpu(source_style_sq_sat, style_level, PREP_GRAY_SQR);
    }

    auto source_style_sat_acc = source_style_sat.packed_accessor64<double, 2>();
    auto source_style_sq_sat_acc = source_style_sq_sat.packed_accessor64<double, 2>();

    torch::Tensor target_style_sat = torch::empty({target_h, target_w}, sat_options);
    torch::Tensor target_style_sq_sat = torch::empty({target_h, target_w}, sat_options);
    auto target_style_sat_acc = target_style_sat.packed_accessor64<double, 2>();
    auto target_style_sq_sat_acc = target_style_sq_sat.packed_accessor64<double, 2>();

    // Initial voting
    krnlVoteWeighted_cpu(target_style_temp_acc, source_style_acc, nnf_acc, error_acc,
                         patch_size, target_h, target_w);
    target_style_prev.copy_(target_style_temp);

    // Random number generator - use fixed seed for deterministic results
    std::mt19937 rng(1337);

    // Main iteration loop
    for (int iter = 0; iter < num_search_vote_iters; ++iter) {
        if (cost_function_mode == COST_FUNCTION_NCC) {
            compute_integral_image_cpu(target_style_sat, target_style_prev, PREP_GRAY);
            compute_integral_image_cpu(target_style_sq_sat, target_style_prev, PREP_GRAY_SQR);
        }

        compute_initial_error_cpu(nnf_acc, error_acc, source_style_acc, target_style_prev_acc,
                                  source_guide_acc, target_guide_acc, target_modulation_guide_acc,
                                  use_modulation, patch_size, style_weights_acc, guide_weights_acc,
                                  cost_function_mode, target_h, target_w);

        // Propagation steps
        for (int i = 0; i < num_patch_match_iters; ++i) {
            propagation_step_cpu(nnf_acc, error_acc, omega_acc, source_style_acc, target_style_prev_acc,
                                 source_guide_acc, target_guide_acc, target_modulation_guide_acc,
                                 use_modulation, style_weights_acc, guide_weights_acc, patch_size,
                                 (i % 2 == 1), uniformity_weight, mask_acc, cost_function_mode,
                                 source_style_sat_acc, source_style_sq_sat_acc, target_style_sat_acc,
                                 target_style_sq_sat_acc, target_h, target_w);
        }

        // Random search
        random_search_step_cpu(nnf_acc, error_acc, omega_acc, source_style_acc, target_style_prev_acc,
                               source_guide_acc, target_guide_acc, target_modulation_guide_acc,
                               use_modulation, style_weights_acc, guide_weights_acc, patch_size,
                               std::max(source_w, source_h) / 2, uniformity_weight, rng, mask_acc,
                               search_pruning_threshold, cost_function_mode, source_style_sat_acc,
                               source_style_sq_sat_acc, target_style_sat_acc, target_style_sq_sat_acc,
                               target_h, target_w);

        // Voting
        if (vote_mode == EBSYNTH_VOTEMODE_WEIGHTED) {
            krnlVoteWeighted_cpu(target_style_temp_acc, source_style_acc, nnf_acc, error_acc,
                                 patch_size, target_h, target_w);
        } else {
            krnlVotePlain_cpu(target_style_temp_acc, source_style_acc, nnf_acc, patch_size,
                              target_h, target_w);
        }

        // Update mask for next iteration
        if (iter < num_search_vote_iters - 1) {
            eval_mask_cpu(mask_acc, target_style_temp_acc, target_style_prev_acc, stop_threshold,
                          target_h, target_w);
            dilate_mask_cpu(mask2_acc, mask_acc, patch_size, target_h, target_w);
            mask.copy_(mask2);
            mask_acc = mask.packed_accessor32<uint8_t, 2>();
            mask2_acc = mask2.packed_accessor32<uint8_t, 2>();
        }

        target_style_prev.copy_(target_style_temp);
    }

    output_image.copy_(target_style_temp);

    compute_initial_error_cpu(nnf_acc, error_acc, source_style_acc, target_style_temp_acc,
                              source_guide_acc, target_guide_acc, target_modulation_guide_acc,
                              use_modulation, patch_size, style_weights_acc, guide_weights_acc,
                              cost_function_mode, target_h, target_w);
}

// ===================================================================
//                RNG STATE INITIALIZER (CPU - no-op for compatibility)
// ===================================================================
void init_rand_states_cpu(torch::Tensor rand_states_tensor) {
    // CPU implementation doesn't need explicit RNG state initialization
    // as we use std::mt19937 which is initialized per-thread
    // This function exists for API compatibility with CUDA version
}
