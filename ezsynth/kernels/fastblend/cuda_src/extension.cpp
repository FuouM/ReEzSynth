#include <torch/extension.h>
#include <vector>
#include "dispatch.h" // Include our dispatch header

// PyTorch CUDA extension for FastBlend operations

// Python-facing remapping function
torch::Tensor fastblend_remap(
    torch::Tensor source_style,
    torch::Tensor nnf,
    int patch_size,
    int pad_size
) {
    TORCH_CHECK(source_style.is_cuda(), "source_style must be a CUDA tensor");
    TORCH_CHECK(nnf.is_cuda(), "nnf must be a CUDA tensor");
    TORCH_CHECK(source_style.is_contiguous(), "source_style must be contiguous");
    TORCH_CHECK(nnf.is_contiguous(), "nnf must be contiguous");
    TORCH_CHECK(source_style.dtype() == torch::kFloat32, "source_style must be float32");
    TORCH_CHECK(nnf.dtype() == torch::kInt32, "nnf must be int32");

    const int batch_size = source_style.size(0);
    const int height = source_style.size(1) - 2 * pad_size;
    const int width = source_style.size(2) - 2 * pad_size;

    TORCH_CHECK(nnf.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(nnf.size(1) == height, "height mismatch");
    TORCH_CHECK(nnf.size(2) == width, "width mismatch");
    TORCH_CHECK(nnf.size(3) == 2, "nnf must have shape (B, H, W, 2)");

    auto target_style = torch::zeros_like(source_style);

    // Dispatch to CUDA implementation
    fastblend_cuda_remap(target_style, source_style, nnf, patch_size, pad_size);

    return target_style;
}

// Python-facing patch error function
torch::Tensor fastblend_patch_error(
    torch::Tensor source,
    torch::Tensor nnf,
    torch::Tensor target,
    int patch_size,
    int pad_size
) {
    TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    TORCH_CHECK(nnf.is_cuda(), "nnf must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(source.is_contiguous(), "source must be contiguous");
    TORCH_CHECK(nnf.is_contiguous(), "nnf must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(source.dtype() == torch::kFloat32, "source must be float32");
    TORCH_CHECK(nnf.dtype() == torch::kInt32, "nnf must be int32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");

    const int batch_size = source.size(0);
    const int height = source.size(1) - 2 * pad_size;
    const int width = source.size(2) - 2 * pad_size;

    TORCH_CHECK(nnf.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(nnf.size(1) == height, "height mismatch");
    TORCH_CHECK(nnf.size(2) == width, "width mismatch");
    TORCH_CHECK(nnf.size(3) == 2, "nnf must have shape (B, H, W, 2)");

    auto error = torch::zeros({batch_size, height, width}, torch::dtype(torch::kFloat32).device(source.device()));

    // Dispatch to CUDA implementation
    fastblend_cuda_patch_error(error, source, nnf, target, patch_size, pad_size);

    return error;
}

// Python-facing pairwise patch error function
torch::Tensor fastblend_pairwise_patch_error(
    torch::Tensor source_a,
    torch::Tensor nnf_a,
    torch::Tensor source_b,
    torch::Tensor nnf_b,
    int patch_size,
    int pad_size
) {
    TORCH_CHECK(source_a.is_cuda(), "source_a must be a CUDA tensor");
    TORCH_CHECK(nnf_a.is_cuda(), "nnf_a must be a CUDA tensor");
    TORCH_CHECK(source_b.is_cuda(), "source_b must be a CUDA tensor");
    TORCH_CHECK(nnf_b.is_cuda(), "nnf_b must be a CUDA tensor");
    TORCH_CHECK(source_a.is_contiguous(), "source_a must be contiguous");
    TORCH_CHECK(nnf_a.is_contiguous(), "nnf_a must be contiguous");
    TORCH_CHECK(source_b.is_contiguous(), "source_b must be contiguous");
    TORCH_CHECK(nnf_b.is_contiguous(), "nnf_b must be contiguous");
    TORCH_CHECK(source_a.dtype() == torch::kFloat32, "source_a must be float32");
    TORCH_CHECK(nnf_a.dtype() == torch::kInt32, "nnf_a must be int32");
    TORCH_CHECK(source_b.dtype() == torch::kFloat32, "source_b must be float32");
    TORCH_CHECK(nnf_b.dtype() == torch::kInt32, "nnf_b must be int32");

    const int batch_size = source_a.size(0);
    const int height = source_a.size(1) - 2 * pad_size;
    const int width = source_a.size(2) - 2 * pad_size;

    TORCH_CHECK(nnf_a.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(nnf_a.size(1) == height, "height mismatch");
    TORCH_CHECK(nnf_a.size(2) == width, "width mismatch");
    TORCH_CHECK(nnf_a.size(3) == 2, "nnf_a must have shape (B, H, W, 2)");
    TORCH_CHECK(nnf_b.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(nnf_b.size(1) == height, "height mismatch");
    TORCH_CHECK(nnf_b.size(2) == width, "width mismatch");
    TORCH_CHECK(nnf_b.size(3) == 2, "nnf_b must have shape (B, H, W, 2)");

    auto error = torch::zeros({batch_size, height, width}, torch::dtype(torch::kFloat32).device(source_a.device()));

    // Dispatch to CUDA implementation
    fastblend_cuda_pairwise_patch_error(error, source_a, nnf_a, source_b, nnf_b, patch_size, pad_size);

    return error;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("remap", &fastblend_remap, "Apply NNF remapping for FastBlend",
          py::arg("source_style"),
          py::arg("nnf"),
          py::arg("patch_size"),
          py::arg("pad_size"));

    m.def("patch_error", &fastblend_patch_error, "Compute patch error for FastBlend",
          py::arg("source"),
          py::arg("nnf"),
          py::arg("target"),
          py::arg("patch_size"),
          py::arg("pad_size"));

    m.def("pairwise_patch_error", &fastblend_pairwise_patch_error, "Compute pairwise patch error for FastBlend",
          py::arg("source_a"),
          py::arg("nnf_a"),
          py::arg("source_b"),
          py::arg("nnf_b"),
          py::arg("patch_size"),
          py::arg("pad_size"));
}