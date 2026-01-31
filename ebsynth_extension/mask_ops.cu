// ReEzSynth/ebsynth_extension/mask_ops.cu
#include "mask_ops.h"

#include <cstdlib>

// ===================================================================
//                        MASKING KERNELS
// ===================================================================

__global__ void eval_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mask.size(1) || y >= mask.size(0)) return;

    const int num_channels = style1.size(2);
    int max_diff = 0;
    for (int c = 0; c < num_channels; ++c) {
        int diff = abs((int)style1[y][x][c] - (int)style2[y][x][c]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    mask[y][x] = (max_diff < stop_threshold) ? 0 : 255;
}

__global__ void dilate_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mask_out.size(1) || y >= mask_out.size(0)) return;

    const int r = patch_size / 2;
    uint8_t msk_val = 0;

    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int nx = x + px;
            int ny = y + py;
            if (nx >= 0 && nx < mask_in.size(1) && ny >= 0 && ny < mask_in.size(0)) {
                if (mask_in[ny][nx] == 255) {
                    msk_val = 255;
                    break;
                }
            }
        }
        if (msk_val == 255) break;
    }
    mask_out[y][x] = msk_val;
}
