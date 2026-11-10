// ReEzSynth/ebsynth_extension/cuda_dispatch.cpp
//
// PyTorch-facing CUDA entry points (compiled by the host compiler, only in CUDA
// builds). Validates tensors, selects the device, allocates scratch on PyTorch's
// caching allocator and enqueues the torch-free kernels (dispatch.cu) on PyTorch's
// current stream. Launch errors are raised as c10::Error via C10_CUDA_CHECK.
#include "dispatch.h"

#include "cuda_launch.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace
{

template <typename T, int N>
ebs::PackedAccessor32<T, N> acc32(const torch::Tensor &t)
{
    // packed_accessor32 validates dtype, rank and 32-bit indexability.
    const auto a = t.packed_accessor32<T, N>();
    return ebs::PackedAccessor32<T, N>(a.data(), t.sizes().data(), t.strides().data());
}

template <typename T, int N>
ebs::PackedAccessor64<T, N> acc64(const torch::Tensor &t)
{
    const auto a = t.packed_accessor64<T, N>();
    return ebs::PackedAccessor64<T, N>(a.data(), t.sizes().data(), t.strides().data());
}

// Kernels index through strides, so only device placement is required here.
void check_device(const torch::Tensor &t, const torch::Device &device, const char *name)
{
    TORCH_CHECK(t.device() == device, name, " must be on ", device, " but is on ", t.device());
}

} // namespace

void ebsynth_cuda_run_level(
    torch::Tensor output_image, torch::Tensor output_error,
    torch::Tensor nnf,
    torch::Tensor style_level, torch::Tensor source_guide_level,
    torch::Tensor target_guide_level, torch::Tensor target_modulation_level,
    torch::Tensor style_weights, torch::Tensor guide_weights,
    float uniformity_weight, int patch_size, int vote_mode,
    int num_search_vote_iters, int num_patch_match_iters,
    int stop_threshold, torch::Tensor rand_states_tensor,
    float search_pruning_threshold, int cost_function_mode,
    bool use_optimization,
    bool use_bilateral,
    float sigma_spatial,
    float sigma_color,
    int n_size_step)
{
    const torch::Device device = style_level.device();
    const at::cuda::OptionalCUDAGuard guard(device);

    const bool use_modulation = target_modulation_level.numel() > 0;
    check_device(output_image, device, "output_image");
    // run_level_cuda copies whole images with cudaMemcpyAsync.
    TORCH_CHECK(output_image.is_contiguous(), "output_image must be contiguous");
    check_device(output_error, device, "output_error");
    check_device(nnf, device, "nnf");
    check_device(style_level, device, "style_level");
    check_device(source_guide_level, device, "source_guide_level");
    check_device(target_guide_level, device, "target_guide_level");
    if (use_modulation)
        check_device(target_modulation_level, device, "target_modulation_level");
    check_device(style_weights, device, "style_weights");
    check_device(guide_weights, device, "guide_weights");
    check_device(rand_states_tensor, device, "rand_states_tensor");
    TORCH_CHECK(rand_states_tensor.is_contiguous(), "rand_states_tensor must be contiguous");

    const int64_t source_h = style_level.size(0);
    const int64_t source_w = style_level.size(1);
    const int64_t target_h = target_guide_level.size(0);
    const int64_t target_w = target_guide_level.size(1);
    TORCH_CHECK(rand_states_tensor.nbytes() >= static_cast<size_t>(target_h * target_w) * ebs::curand_state_size(),
                "rand_states_tensor holds fewer than target_h * target_w curandState entries");

    const auto int_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto u8_options = torch::TensorOptions().device(device).dtype(torch::kUInt8);
    const auto f64_options = torch::TensorOptions().device(device).dtype(torch::kFloat64);

    torch::Tensor omega_map = torch::zeros({source_h, source_w}, int_options);
    torch::Tensor style_temp = torch::zeros_like(output_image);
    torch::Tensor style_prev = torch::zeros_like(output_image);
    torch::Tensor mask = torch::full({target_h, target_w}, 255, u8_options);
    torch::Tensor mask2 = torch::empty_like(mask);

    ebs::LevelBuffers b{};
    b.nnf = acc32<int32_t, 3>(nnf);
    b.error = acc32<float, 2>(output_error);
    b.output = acc32<uint8_t, 3>(output_image);
    b.source_style = acc32<uint8_t, 3>(style_level);
    b.source_guide = acc32<uint8_t, 3>(source_guide_level);
    b.target_guide = acc32<uint8_t, 3>(target_guide_level);
    b.target_modulation = use_modulation ? acc32<uint8_t, 3>(target_modulation_level) : b.source_guide;
    b.use_modulation = use_modulation;
    b.style_weights = acc32<float, 1>(style_weights);
    b.guide_weights = acc32<float, 1>(guide_weights);
    b.omega = acc32<int32_t, 2>(omega_map);
    b.style_temp = acc32<uint8_t, 3>(style_temp);
    b.style_prev = acc32<uint8_t, 3>(style_prev);
    b.mask = acc32<uint8_t, 2>(mask);
    b.mask2 = acc32<uint8_t, 2>(mask2);
    b.rand_states = rand_states_tensor.data_ptr();

    // Summed-area tables are only read by the NCC cost function.
    torch::Tensor source_sat, source_sq_sat, target_sat, target_sq_sat, sat_scratch;
    if (cost_function_mode == 1) // COST_FUNCTION_NCC (cost_functions.h)
    {
        source_sat = torch::empty({source_h, source_w}, f64_options);
        source_sq_sat = torch::empty({source_h, source_w}, f64_options);
        target_sat = torch::empty({target_h, target_w}, f64_options);
        target_sq_sat = torch::empty({target_h, target_w}, f64_options);
        sat_scratch = torch::empty({std::max(source_h * source_w, target_h * target_w)}, f64_options);
        b.source_sat = acc64<double, 2>(source_sat);
        b.source_sq_sat = acc64<double, 2>(source_sq_sat);
        b.target_sat = acc64<double, 2>(target_sat);
        b.target_sq_sat = acc64<double, 2>(target_sq_sat);
        b.sat_scratch = sat_scratch.data_ptr<double>();
    }

    ebs::LevelParams p{};
    p.uniformity_weight = uniformity_weight;
    p.patch_size = patch_size;
    p.vote_mode = vote_mode;
    p.num_search_vote_iters = num_search_vote_iters;
    p.num_patch_match_iters = num_patch_match_iters;
    p.stop_threshold = stop_threshold;
    p.search_pruning_threshold = search_pruning_threshold;
    p.cost_function_mode = cost_function_mode;
    p.use_bilateral = use_bilateral;
    p.sigma_spatial = sigma_spatial;
    p.sigma_color = sigma_color;
    p.n_size_step = n_size_step;

    // Scratch tensors are released back to the caching allocator on return; that is
    // safe because the allocator orders reuse after work already queued on this stream.
    C10_CUDA_CHECK(ebs::run_level_cuda(b, p, at::cuda::getCurrentCUDAStream()));
}

void init_rand_states_cuda(torch::Tensor rand_states_tensor)
{
    TORCH_CHECK(rand_states_tensor.is_contiguous(), "rand_states_tensor must be contiguous");
    const at::cuda::OptionalCUDAGuard guard(rand_states_tensor.device());
    const int64_t num_states = static_cast<int64_t>(rand_states_tensor.nbytes() / ebs::curand_state_size());
    TORCH_CHECK(num_states <= INT32_MAX, "too many curandState entries");
    C10_CUDA_CHECK(ebs::init_rand_states_cuda_launch(
        rand_states_tensor.data_ptr(), num_states, at::cuda::getCurrentCUDAStream()));
}
