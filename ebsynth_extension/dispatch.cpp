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
    int cost_function_mode,
    bool use_optimization,
    bool use_bilateral,
    float sigma_spatial,
    float sigma_color,
    int n_size_step)
{

    // Auto-detect device type from input tensor
    if (style_level.device().is_cpu())
    {
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
            cost_function_mode,
            use_optimization,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step);
    }
    else
    {
#ifndef CPU_ONLY
        // Route to CUDA implementation
        // ebsynth_cuda_run_level is declared in dispatch.h usually
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
            cost_function_mode,
            use_optimization,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step);
#else
        throw std::runtime_error("ebsynth_extension was compiled without CUDA support.");
#endif
    }
}

// ===================================================================
//                UNIFIED RNG INITIALIZER
// ===================================================================
void init_rand_states(torch::Tensor rand_states_tensor)
{
    if (rand_states_tensor.device().is_cpu())
    {
        // CPU RNG initialization (no-op for compatibility)
        init_rand_states_cpu(rand_states_tensor);
    }
    else
    {
#ifndef CPU_ONLY
        init_rand_states_cuda(rand_states_tensor);
#else
        throw std::runtime_error("ebsynth_extension was compiled without CUDA support.");
#endif
    }
}
