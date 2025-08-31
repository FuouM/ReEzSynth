// ReEzSynth/ebsynth_extension/ebsynth.h
#pragma once
#include <torch/extension.h>
#include <vector>

// CUDA function for a single synthesis level
void ebsynth_cuda_run_level(
    torch::Tensor output_image,
    torch::Tensor output_error,
    torch::Tensor nnf, // NNF is now input and output
    torch::Tensor style_level,
    torch::Tensor source_guide_level,
    torch::Tensor target_guide_level,
    torch::Tensor target_modulation_level, // New: Modulation map
    torch::Tensor style_weights,
    torch::Tensor guide_weights,
    float uniformity_weight,
    int patch_size,
    int vote_mode,
    int num_search_vote_iters,
    int num_patch_match_iters,
    int stop_threshold,
    torch::Tensor rand_states_tensor);

// CUDA function to initialize random states
void init_rand_states_cuda(torch::Tensor rand_states_tensor);