// FastBlend CUDA dispatch functions
#include "dispatch.h"
#include "kernels.h"

// CUDA dispatch function for remapping
void fastblend_cuda_remap(
    torch::Tensor target_style,
    torch::Tensor source_style,
    torch::Tensor nnf,
    int patch_size,
    int pad_size
) {
    const int batch_size = source_style.size(0);
    const int height = source_style.size(1) - 2 * pad_size;
    const int width = source_style.size(2) - 2 * pad_size;
    const int channel = source_style.size(3);

    launch_remap_kernel(
        height, width, channel, patch_size, pad_size,
        source_style.data_ptr<float>(),
        nnf.data_ptr<int>(),
        target_style.data_ptr<float>(),
        batch_size
    );
}

// CUDA dispatch function for patch error
void fastblend_cuda_patch_error(
    torch::Tensor error,
    torch::Tensor source,
    torch::Tensor nnf,
    torch::Tensor target,
    int patch_size,
    int pad_size
) {
    const int batch_size = source.size(0);
    const int height = source.size(1) - 2 * pad_size;
    const int width = source.size(2) - 2 * pad_size;
    const int channel = source.size(3);

    launch_patch_error_kernel(
        height, width, channel, patch_size, pad_size,
        source.data_ptr<float>(),
        nnf.data_ptr<int>(),
        target.data_ptr<float>(),
        error.data_ptr<float>(),
        batch_size
    );
}

// CUDA dispatch function for pairwise patch error
void fastblend_cuda_pairwise_patch_error(
    torch::Tensor error,
    torch::Tensor source_a,
    torch::Tensor nnf_a,
    torch::Tensor source_b,
    torch::Tensor nnf_b,
    int patch_size,
    int pad_size
) {
    const int batch_size = source_a.size(0);
    const int height = source_a.size(1) - 2 * pad_size;
    const int width = source_a.size(2) - 2 * pad_size;
    const int channel = source_a.size(3);

    launch_pairwise_patch_error_kernel(
        height, width, channel, patch_size, pad_size,
        source_a.data_ptr<float>(),
        nnf_a.data_ptr<int>(),
        source_b.data_ptr<float>(),
        nnf_b.data_ptr<int>(),
        error.data_ptr<float>(),
        batch_size
    );
}