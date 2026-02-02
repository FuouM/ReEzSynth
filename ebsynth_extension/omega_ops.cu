// ReEzSynth/ebsynth_extension/omega_ops.cu
#include "omega_ops.h"

// ===================================================================
//                        UNIFORMITY KERNELS
// ===================================================================

__device__ void update_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec)
{
    const int r = patch_size / 2;
    for (int py = -r; py <= r; ++py)
    {
        for (int px = -r; px <= r; ++px)
        {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_map.size(1) && cur_sy >= 0 && cur_sy < omega_map.size(0))
            {
                atomicAdd(&omega_map[cur_sy][cur_sx], incdec);
            }
        }
    }
}

__device__ float patch_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size)
{
    const int r = patch_size / 2;
    float sum = 0;
    for (int py = -r; py <= r; ++py)
    {
        for (int px = -r; px <= r; ++px)
        {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_map.size(1) && cur_sy >= 0 && cur_sy < omega_map.size(0))
            {
                sum += omega_map[cur_sy][cur_sx];
            }
        }
    }
    return sum;
}

__global__ void populate_initial_omega_kernel(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= nnf.size(1) || ty >= nnf.size(0))
        return;

    int sx = nnf[ty][tx][0];
    int sy = nnf[ty][tx][1];
    update_omega(omega_map, sx, sy, patch_size, 1);
}
