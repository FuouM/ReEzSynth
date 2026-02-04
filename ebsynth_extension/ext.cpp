#include <torch/extension.h>
#include <vector>

#include "dispatch.h" // Include our unified dispatch header

// Python-facing function to run a single pyramid level
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_level(
    torch::Tensor style_level,
    torch::Tensor source_guide_level,
    torch::Tensor target_guide_level,
    torch::Tensor target_modulation_level, // New: Modulation map
    torch::Tensor nnf,                     // NNF is now an input
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
    bool use_optimization) // New parameter
{
    // Input validation (removed CUDA-only checks to support CPU)
    TORCH_CHECK(style_level.is_contiguous(), "Style tensor must be contiguous");
    TORCH_CHECK(style_level.scalar_type() == torch::kUInt8, "Style tensor must be uint8");
    TORCH_CHECK(style_level.device() == nnf.device(), "Style and NNF tensors must be on the same device");

    // Prepare output tensors on the same device as input
    auto options = torch::TensorOptions().device(style_level.device()).dtype(torch::kUInt8);
    auto error_options = torch::TensorOptions().device(style_level.device()).dtype(torch::kFloat32);

    const int target_h = target_guide_level.size(0);
    const int target_w = target_guide_level.size(1);
    const int num_style_channels = style_level.size(2);

    torch::Tensor output_image = torch::zeros({target_h, target_w, num_style_channels}, options);
    torch::Tensor output_error = torch::zeros({target_h, target_w}, error_options);

    // Dispatch to the appropriate implementation (CPU or CUDA)
    ebsynth_run_level(
        output_image,
        output_error,
        nnf, // Pass NNF to be modified in-place
        style_level,
        source_guide_level,
        target_guide_level,
        target_modulation_level, // Pass modulation map
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
        use_optimization); // Pass new parameter

    // Return the results including the modified NNF
    return {output_image, output_error, nnf};
}

// Python-facing function to initialize RNG states once
void init_rand_states_wrapper(torch::Tensor rand_states_tensor)
{
    // Dispatch to appropriate implementation based on device
    init_rand_states(rand_states_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run_level", &run_level, "Run one level of Ebsynth on PyTorch Tensors (CPU or CUDA)",
          py::arg("style_level"),
          py::arg("source_guide_level"),
          py::arg("target_guide_level"),
          py::arg("target_modulation_level"),
          py::arg("nnf"),
          py::arg("style_weights"),
          py::arg("guide_weights"),
          py::arg("uniformity_weight"),
          py::arg("patch_size"),
          py::arg("vote_mode"),
          py::arg("num_search_vote_iters"),
          py::arg("num_patch_match_iters"),
          py::arg("stop_threshold"),
          py::arg("rand_states_tensor"),
          py::arg("search_pruning_threshold"),
          py::arg("cost_function_mode"),
          py::arg("use_optimization"));
    m.def("init_rand_states", &init_rand_states_wrapper, "Initialize random number generator states (CPU or CUDA)");
}