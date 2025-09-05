// ReEzSynth/ebsynth_extension/dispatch.cu
#include "dispatch.h"
#include "kernels.h"

#include <stdexcept>

// ===================================================================
//                MAIN DISPATCH FUNCTION (SINGLE LEVEL)
// ===================================================================
void ebsynth_cuda_run_level(
    torch::Tensor output_image, torch::Tensor output_error,
    torch::Tensor nnf,
    torch::Tensor style_level, torch::Tensor source_guide_level,
    torch::Tensor target_guide_level, torch::Tensor target_modulation_level,
    torch::Tensor style_weights, torch::Tensor guide_weights,
    float uniformity_weight, int patch_size, int vote_mode,
    int num_search_vote_iters, int num_patch_match_iters,
    int stop_threshold, torch::Tensor rand_states_tensor,
    float search_pruning_threshold, int cost_function_mode) {
    
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
    auto target_modulation_guide_acc = use_modulation ? target_modulation_level.packed_accessor32<uint8_t, 3>() : source_guide_acc; // dummy if not used
    
    auto style_weights_acc = style_weights.packed_accessor32<float, 1>();
    auto guide_weights_acc = guide_weights.packed_accessor32<float, 1>();
    
    const dim3 threads(16, 16);
    const dim3 blocks((target_w + threads.x - 1) / threads.x, (target_h + threads.y - 1) / threads.y);
    
    torch::Tensor omega_map = torch::zeros({source_h, source_w}, torch::kInt32).to(style_level.device());
    auto omega_acc = omega_map.packed_accessor32<int32_t, 2>();
    populate_initial_omega_kernel<<<blocks, threads>>>(omega_acc, nnf_acc, patch_size);
    
    torch::Tensor target_style_temp = torch::zeros_like(output_image);
    torch::Tensor target_style_prev = torch::zeros_like(output_image);
    torch::Tensor mask = torch::full({target_h, target_w}, 255, torch::kUInt8).to(style_level.device());
    torch::Tensor mask2 = torch::empty_like(mask);

    auto target_style_temp_acc = target_style_temp.packed_accessor32<uint8_t, 3>();
    auto target_style_prev_acc = target_style_prev.packed_accessor32<uint8_t, 3>();
    auto mask_acc = mask.packed_accessor32<uint8_t, 2>();
    auto mask2_acc = mask2.packed_accessor32<uint8_t, 2>();
    
    krnlVoteWeighted<<<blocks, threads>>>(target_style_temp_acc, source_style_acc, nnf_acc, error_acc, patch_size);
    target_style_prev.copy_(target_style_temp);
    
    curandState* rand_states = (curandState*)rand_states_tensor.data_ptr();

    for (int iter = 0; iter < num_search_vote_iters; ++iter) {
        compute_initial_error_kernel<<<blocks, threads>>>(nnf_acc, error_acc, source_style_acc, target_style_prev_acc, source_guide_acc, target_guide_acc, target_modulation_guide_acc, use_modulation, patch_size, style_weights_acc, guide_weights_acc, cost_function_mode);

        // --- REVERTED PROPAGATION LOGIC ---
        for (int i = 0; i < num_patch_match_iters; ++i) {
            propagation_step_kernel<<<blocks, threads>>>(nnf_acc, error_acc, omega_acc, source_style_acc, target_style_prev_acc, source_guide_acc, target_guide_acc, target_modulation_guide_acc, use_modulation, style_weights_acc, guide_weights_acc, patch_size, (i % 2 == 1), uniformity_weight, mask_acc, cost_function_mode);
        }
        // --- END REVERTED LOGIC ---

        random_search_step_kernel<<<blocks, threads>>>(nnf_acc, error_acc, omega_acc, source_style_acc, target_style_prev_acc, source_guide_acc, target_guide_acc, target_modulation_guide_acc, use_modulation, style_weights_acc, guide_weights_acc, patch_size, std::max(source_w, source_h) / 2, uniformity_weight, rand_states, mask_acc, search_pruning_threshold, cost_function_mode);

        if (vote_mode == EBSYNTH_VOTEMODE_WEIGHTED) {
            krnlVoteWeighted<<<blocks, threads>>>(target_style_temp_acc, source_style_acc, nnf_acc, error_acc, patch_size);
        } else {
            krnlVotePlain<<<blocks, threads>>>(target_style_temp_acc, source_style_acc, nnf_acc, patch_size);
        }
        
        if (iter < num_search_vote_iters - 1) {
            eval_mask_kernel<<<blocks, threads>>>(mask_acc, target_style_temp_acc, target_style_prev_acc, stop_threshold);
            dilate_mask_kernel<<<blocks, threads>>>(mask2_acc, mask_acc, patch_size);
            std::swap(mask, mask2);
            mask_acc = mask.packed_accessor32<uint8_t, 2>();
            mask2_acc = mask2.packed_accessor32<uint8_t, 2>();
        }
        
        target_style_prev.copy_(target_style_temp);
    }
    
    output_image.copy_(target_style_temp);
    
    compute_initial_error_kernel<<<blocks, threads>>>(nnf_acc, error_acc, source_style_acc, target_style_temp_acc, source_guide_acc, target_guide_acc, target_modulation_guide_acc, use_modulation, patch_size, style_weights_acc, guide_weights_acc, cost_function_mode);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}


// ===================================================================
//                RNG STATE INITIALIZER
// ===================================================================
void init_rand_states_cuda(torch::Tensor rand_states_tensor) {
    curandState* states_ptr = (curandState*)rand_states_tensor.data_ptr();
    int num_states = rand_states_tensor.numel() * rand_states_tensor.element_size() / sizeof(curandState);
    
    const int threads_per_block = 256;
    const int num_blocks = (num_states + threads_per_block - 1) / threads_per_block;
    
    init_rand_states_kernel<<<num_blocks, threads_per_block>>>(states_ptr, num_states);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}