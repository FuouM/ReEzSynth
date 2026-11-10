// ReEzSynth/ezsynth/ebsynth_extension/dispatch.cu
//
// Kernel orchestration for one synthesis level. No PyTorch headers here (see
// accessor.h); the PyTorch-facing entry points live in cuda_dispatch.cpp.
// All work is enqueued on the caller's stream (PyTorch's current stream), so it is
// ordered with surrounding tensor ops and needs no host synchronization.
#include "cuda_launch.h"
#include "kernels.h"

#include <algorithm>
#include <utility>

#define EBS_RETURN_IF_ERROR(expr)      \
    do                                 \
    {                                  \
        const cudaError_t _err = expr; \
        if (_err != cudaSuccess)       \
            return _err;               \
    } while (0)

// Launch-configuration errors are reported by cudaGetLastError right after a launch.
#define EBS_LAUNCH_CHECK() EBS_RETURN_IF_ERROR(cudaGetLastError())

namespace ebs
{

namespace
{
cudaError_t copy_image(PackedAccessor32<uint8_t, 3> dst, PackedAccessor32<uint8_t, 3> src, cudaStream_t stream)
{
    const std::size_t bytes = static_cast<std::size_t>(src.size(0)) * src.size(1) * src.size(2);
    return cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, stream);
}
} // namespace

cudaError_t run_level_cuda(LevelBuffers b, const LevelParams &p, cudaStream_t stream)
{
    const int source_h = b.source_style.size(0);
    const int source_w = b.source_style.size(1);
    const int target_h = b.target_guide.size(0);
    const int target_w = b.target_guide.size(1);
    const int patch_size = p.patch_size;
    const bool ncc = p.cost_function_mode == COST_FUNCTION_NCC;

    const dim3 threads(16, 16);
    const dim3 blocks((target_w + threads.x - 1) / threads.x, (target_h + threads.y - 1) / threads.y);

    populate_initial_omega_kernel<<<blocks, threads, 0, stream>>>(b.omega, b.nnf, patch_size);
    EBS_LAUNCH_CHECK();

    // Source SATs are computed once per level; target SATs every iteration.
    if (ncc)
    {
        EBS_RETURN_IF_ERROR(compute_integral_image_cuda(b.source_sat, b.source_style, b.sat_scratch, PREP_GRAY, stream));
        EBS_RETURN_IF_ERROR(compute_integral_image_cuda(b.source_sq_sat, b.source_style, b.sat_scratch, PREP_GRAY_SQR, stream));
    }

    krnlVoteWeighted<<<blocks, threads, 0, stream>>>(b.style_temp, b.source_style, b.nnf, b.error, patch_size);
    EBS_LAUNCH_CHECK();
    EBS_RETURN_IF_ERROR(copy_image(b.style_prev, b.style_temp, stream));

    // One curandState per target pixel, indexed y * target_w + x by the kernels.
    // States persist across calls, so successive levels/frames keep advancing them.
    curandState *rand_states = static_cast<curandState *>(b.rand_states);

    for (int iter = 0; iter < p.num_search_vote_iters; ++iter)
    {
        if (ncc)
        {
            EBS_RETURN_IF_ERROR(compute_integral_image_cuda(b.target_sat, b.style_prev, b.sat_scratch, PREP_GRAY, stream));
            EBS_RETURN_IF_ERROR(compute_integral_image_cuda(b.target_sq_sat, b.style_prev, b.sat_scratch, PREP_GRAY_SQR, stream));
        }

        compute_initial_error_kernel<<<blocks, threads, 0, stream>>>(b.nnf, b.error, b.source_style, b.style_prev, b.source_guide, b.target_guide, b.target_modulation, b.use_modulation, patch_size, b.style_weights, b.guide_weights, p.cost_function_mode, p.use_bilateral, p.sigma_spatial, p.sigma_color, p.n_size_step);
        EBS_LAUNCH_CHECK();

        for (int i = 0; i < p.num_patch_match_iters; ++i)
        {
            propagation_step_kernel<<<blocks, threads, 0, stream>>>(b.nnf, b.error, b.omega, b.source_style, b.style_prev, b.source_guide, b.target_guide, b.target_modulation, b.use_modulation, b.style_weights, b.guide_weights, patch_size, (i % 2 == 1), p.uniformity_weight, b.mask, p.cost_function_mode, b.source_sat, b.source_sq_sat, b.target_sat, b.target_sq_sat, p.use_bilateral, p.sigma_spatial, p.sigma_color, p.n_size_step);
            EBS_LAUNCH_CHECK();
        }

        random_search_step_kernel<<<blocks, threads, 0, stream>>>(b.nnf, b.error, b.omega, b.source_style, b.style_prev, b.source_guide, b.target_guide, b.target_modulation, b.use_modulation, b.style_weights, b.guide_weights, patch_size, std::max(source_w, source_h) / 2, p.uniformity_weight, rand_states, b.mask, p.search_pruning_threshold, p.cost_function_mode, b.source_sat, b.source_sq_sat, b.target_sat, b.target_sq_sat, p.use_bilateral, p.sigma_spatial, p.sigma_color, p.n_size_step);
        EBS_LAUNCH_CHECK();

        if (p.vote_mode == EBSYNTH_VOTEMODE_WEIGHTED)
        {
            krnlVoteWeighted<<<blocks, threads, 0, stream>>>(b.style_temp, b.source_style, b.nnf, b.error, patch_size);
        }
        else
        {
            krnlVotePlain<<<blocks, threads, 0, stream>>>(b.style_temp, b.source_style, b.nnf, patch_size);
        }
        EBS_LAUNCH_CHECK();

        if (iter < p.num_search_vote_iters - 1)
        {
            eval_mask_kernel<<<blocks, threads, 0, stream>>>(b.mask, b.style_temp, b.style_prev, p.stop_threshold);
            EBS_LAUNCH_CHECK();
            dilate_mask_kernel<<<blocks, threads, 0, stream>>>(b.mask2, b.mask, patch_size);
            EBS_LAUNCH_CHECK();
            std::swap(b.mask, b.mask2);
        }

        EBS_RETURN_IF_ERROR(copy_image(b.style_prev, b.style_temp, stream));
    }

    EBS_RETURN_IF_ERROR(copy_image(b.output, b.style_temp, stream));

    compute_initial_error_kernel<<<blocks, threads, 0, stream>>>(b.nnf, b.error, b.source_style, b.style_temp, b.source_guide, b.target_guide, b.target_modulation, b.use_modulation, patch_size, b.style_weights, b.guide_weights, p.cost_function_mode, p.use_bilateral, p.sigma_spatial, p.sigma_color, p.n_size_step);
    EBS_LAUNCH_CHECK();
    return cudaSuccess;
}

std::size_t curand_state_size()
{
    return sizeof(curandState);
}

cudaError_t init_rand_states_cuda_launch(void *states, int64_t num_states, cudaStream_t stream)
{
    if (num_states <= 0)
        return cudaSuccess;
    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((num_states + threads_per_block - 1) / threads_per_block);
    init_rand_states_kernel<<<num_blocks, threads_per_block, 0, stream>>>(static_cast<curandState *>(states), static_cast<int>(num_states));
    return cudaGetLastError();
}

} // namespace ebs
