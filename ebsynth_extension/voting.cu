// ReEzSynth/ebsynth_extension/voting.cu
#include "voting.h"

#include <cmath>

// ===================================================================
//                        VOTING KERNELS
// ===================================================================

__global__ void krnlVotePlain(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= target.size(1) || y >= target.size(0))
        return;

    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    Vec<4, float> sumColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float sumWeight = 0.0f;

    for (int py = -r; py <= r; py++)
    {
        for (int px = -r; px <= r; px++)
        {
            int t_neighbor_x = x - px;
            int t_neighbor_y = y - py;
            if (t_neighbor_x >= 0 && t_neighbor_x < target.size(1) && t_neighbor_y >= 0 && t_neighbor_y < target.size(0))
            {
                int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
                int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

                int source_x = s_neighbor_center_x + px;
                int source_y = s_neighbor_center_y + py;

                if (source_x >= 0 && source_x < source.size(1) && source_y >= 0 && source_y < source.size(0))
                {
                    const float weight = 1.0f;
                    for (int c = 0; c < num_style_channels; ++c)
                    {
                        sumColor.v[c] += weight * (float)source[source_y][source_x][c];
                    }
                    sumWeight += weight;
                }
            }
        }
    }

    if (sumWeight > 0.0001f)
    {
        for (int c = 0; c < num_style_channels; ++c)
        {
            float val = sumColor.v[c] / sumWeight;
            target[y][x][c] = (uint8_t)fminf(fmaxf(val, 0.0f), 255.0f);
        }
    }
}

__global__ void krnlVoteWeighted(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    int patch_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= target.size(1) || y >= target.size(0))
        return;

    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    Vec<4, float> sumColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float sumWeight = 0.0f;

    for (int py = -r; py <= r; py++)
    {
        for (int px = -r; px <= r; px++)
        {
            int t_neighbor_x = x - px;
            int t_neighbor_y = y - py;

            if (t_neighbor_x >= 0 && t_neighbor_x < target.size(1) &&
                t_neighbor_y >= 0 && t_neighbor_y < target.size(0))
            {

                int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
                int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

                int source_x = s_neighbor_center_x + px;
                int source_y = s_neighbor_center_y + py;

                if (source_x >= 0 && source_x < source.size(1) &&
                    source_y >= 0 && source_y < source.size(0))
                {

                    const float error = error_map[t_neighbor_y][t_neighbor_x];
                    const float weight = 1.0f / (1.0f + error);

                    for (int c = 0; c < num_style_channels; ++c)
                    {
                        sumColor.v[c] += weight * (float)source[source_y][source_x][c];
                    }
                    sumWeight += weight;
                }
            }
        }
    }

    if (sumWeight > 0.0001f)
    {
        for (int c = 0; c < num_style_channels; ++c)
        {
            float val = sumColor.v[c] / sumWeight;
            target[y][x][c] = (uint8_t)fminf(fmaxf(val, 0.0f), 255.0f);
        }
    }
}
