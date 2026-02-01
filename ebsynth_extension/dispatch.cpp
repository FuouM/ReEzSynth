// ReEzSynth/ebsynth_extension/dispatch.cpp
#include "dispatch.h"

#include "cpu/dispatch_cpu.h"

#include <stdexcept>

// ===================================================================
//                UNIFIED DISPATCH FUNCTION
// ===================================================================
void ebsynth_run_level(
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

    // Auto-detect device type from input tensor
    if (style_level.device().is_cpu()) {
        // Route to CPU implementation
        ebsynth_cpu_run_level(
            output_image,
            output_error,
            nnf,
            style_level,
            source_guide_level,
            target_guide_level,
            target_modulation_level,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            vote_mode,
            num_search_vote_iters,
            num_patch_match_iters,
            stop_threshold,
            rand_states_tensor,
            search_pruning_threshold,
            cost_function_mode);
    } else {
        // Route to CUDA implementation
        ebsynth_cuda_run_level(
            output_image,
            output_error,
            nnf,
            style_level,
            source_guide_level,
            target_guide_level,
            target_modulation_level,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            vote_mode,
            num_search_vote_iters,
            num_patch_match_iters,
            stop_threshold,
            rand_states_tensor,
            search_pruning_threshold,
            cost_function_mode);
    }
}

// ===================================================================
//                UNIFIED RNG INITIALIZER
// ===================================================================
void init_rand_states(torch::Tensor rand_states_tensor) {
    if (rand_states_tensor.device().is_cpu()) {
        // CPU RNG initialization (no-op for compatibility)
        init_rand_states_cpu(rand_states_tensor);
    } else {
        init_rand_states_cuda(rand_states_tensor);
    }
}
