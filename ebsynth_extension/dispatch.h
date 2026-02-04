// ReEzSynth/ebsynth_extension/dispatch.h
#pragma once
#include <torch/extension.h>

// Unified dispatch function - auto-detects device type and routes to CPU or CUDA
void ebsynth_run_level(
    torch::Tensor output_image,
    torch::Tensor output_error,
    torch::Tensor nnf, // NNF is now input and output
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
    int cost_function_mode,
    bool use_optimization);

// Unified RNG initialization - auto-detects device type
void init_rand_states(torch::Tensor rand_states_tensor);

// CUDA function for a single synthesis level
void ebsynth_cuda_run_level(
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
    int cost_function_mode,
    bool use_optimization);

// CUDA function to initialize random states
void init_rand_states_cuda(torch::Tensor rand_states_tensor);

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
    int cost_function_mode,
    bool use_optimization);

// CPU function to initialize random states (no-op for compatibility)
void init_rand_states_cpu(torch::Tensor rand_states_tensor);