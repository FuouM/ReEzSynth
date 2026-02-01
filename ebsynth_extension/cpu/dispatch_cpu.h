// ReEzSynth/ebsynth_extension/cpu/dispatch_cpu.h
#pragma once

#include <torch/extension.h>

// CPU function for a single synthesis level
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
    int cost_function_mode);

// CPU function to initialize random states (no-op for CPU, but kept for API compatibility)
void init_rand_states_cpu(torch::Tensor rand_states_tensor);
