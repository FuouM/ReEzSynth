// ReEzSynth/ebsynth_extension/cpu/voting_cpu.cpp
#include "voting_cpu.h"

#include <cmath>
#include <algorithm>

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// ===================================================================
//                        VOTING OPERATIONS
// ===================================================================

void krnlVotePlain_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size,
    int target_h, int target_w)
{

    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    const int source_h = source.size(0);
    const int source_w = source.size(1);

// Parallelize over rows for better cache locality
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int y = 0; y < target_h; ++y)
    {
        // Stack-allocated accumulator for better performance
        float sumColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int x = 0; x < target_w; ++x)
        {
            // Reset accumulators
            for (int c = 0; c < num_style_channels; ++c)
            {
                sumColor[c] = 0.0f;
            }
            float sumWeight = 0.0f;

            for (int py = -r; py <= r; py++)
            {
                for (int px = -r; px <= r; px++)
                {
                    int t_neighbor_x = x - px;
                    int t_neighbor_y = y - py;

                    if (t_neighbor_x >= 0 && t_neighbor_x < target_w &&
                        t_neighbor_y >= 0 && t_neighbor_y < target_h)
                    {

                        int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
                        int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

                        int source_x = s_neighbor_center_x + px;
                        int source_y = s_neighbor_center_y + py;

                        if (source_x >= 0 && source_x < source_w &&
                            source_y >= 0 && source_y < source_h)
                        {

                            const float weight = 1.0f;
                            for (int c = 0; c < num_style_channels; ++c)
                            {
                                sumColor[c] += weight * (float)source[source_y][source_x][c];
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
                    float val = sumColor[c] / sumWeight;
                    target[y][x][c] = (uint8_t)std::min(std::max(val, 0.0f), 255.0f);
                }
            }
        }
    }
}

void krnlVoteWeighted_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    int patch_size,
    int target_h, int target_w)
{

    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    const int source_h = source.size(0);
    const int source_w = source.size(1);

// Parallelize over rows for better cache locality
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int y = 0; y < target_h; ++y)
    {
        // Stack-allocated accumulator for better performance
        float sumColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int x = 0; x < target_w; ++x)
        {
            // Reset accumulators
            for (int c = 0; c < num_style_channels; ++c)
            {
                sumColor[c] = 0.0f;
            }
            float sumWeight = 0.0f;

            for (int py = -r; py <= r; py++)
            {
                for (int px = -r; px <= r; px++)
                {
                    int t_neighbor_x = x - px;
                    int t_neighbor_y = y - py;

                    if (t_neighbor_x >= 0 && t_neighbor_x < target_w &&
                        t_neighbor_y >= 0 && t_neighbor_y < target_h)
                    {

                        int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
                        int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

                        int source_x = s_neighbor_center_x + px;
                        int source_y = s_neighbor_center_y + py;

                        if (source_x >= 0 && source_x < source_w &&
                            source_y >= 0 && source_y < source_h)
                        {

                            const float error = error_map[t_neighbor_y][t_neighbor_x];
                            const float weight = 1.0f / (1.0f + error);

                            for (int c = 0; c < num_style_channels; ++c)
                            {
                                sumColor[c] += weight * (float)source[source_y][source_x][c];
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
                    float val = sumColor[c] / sumWeight;
                    target[y][x][c] = (uint8_t)std::min(std::max(val, 0.0f), 255.0f);
                }
            }
        }
    }
}
