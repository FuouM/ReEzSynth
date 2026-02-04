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
void krnlVoteIncremental_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf_current,
    torch::PackedTensorAccessor32<int32_t, 3> nnf_prev,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<float, 3> accumulators,
    int patch_size,
    int target_h, int target_w)
{
    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    const int source_h = source.size(0);
    const int source_w = source.size(1);

    // 1. Identify changed patches and update accumulators
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int ty = 0; ty < target_h; ++ty)
    {
        for (int tx = 0; tx < target_w; ++tx)
        {
            int sx_curr = nnf_current[ty][tx][0];
            int sy_curr = nnf_current[ty][tx][1];
            int sx_prev = nnf_prev[ty][tx][0];
            int sy_prev = nnf_prev[ty][tx][1];

            if (sx_curr != sx_prev || sy_curr != sy_prev)
            {
                // Subtract old contribution
                const float error_prev = 0.0f;  // Simplified for now, or we'd need error_map_prev
                const float weight_prev = 1.0f; // Simplified

                // Add new contribution
                const float error_curr = error_map[ty][tx];
                const float weight_curr = 1.0f / (1.0f + error_curr);

                for (int py = -r; py <= r; py++)
                {
                    for (int px = -r; px <= r; px++)
                    {
                        int target_x = tx + px;
                        int target_y = ty + py;

                        if (target_x >= 0 && target_x < target_w && target_y >= 0 && target_y < target_h)
                        {
                            int s_x_old = sx_prev + px;
                            int s_y_old = sy_prev + py;
                            int s_x_new = sx_curr + px;
                            int s_y_new = sy_curr + py;

                            // Actual pixel coordinates in source
                            if (s_x_old >= 0 && s_x_old < source_w && s_y_old >= 0 && s_y_old < source_h)
                            {
                                for (int c = 0; c < num_style_channels; ++c)
                                {
#pragma omp atomic
                                    accumulators[target_y][target_x][c] -= weight_prev * (float)source[s_y_old][s_x_old][c];
                                }
#pragma omp atomic
                                accumulators[target_y][target_x][num_style_channels] -= weight_prev;
                            }

                            if (s_x_new >= 0 && s_x_new < source_w && s_y_new >= 0 && s_y_new < source_h)
                            {
                                for (int c = 0; c < num_style_channels; ++c)
                                {
#pragma omp atomic
                                    accumulators[target_y][target_x][c] += weight_curr * (float)source[s_y_new][s_x_new][c];
                                }
#pragma omp atomic
                                accumulators[target_y][target_x][num_style_channels] += weight_curr;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Re-normalize pixels (can be optimized to only "dirty" pixels, but full pass is fast)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < target_h; ++y)
    {
        for (int x = 0; x < target_w; ++x)
        {
            float sumWeight = accumulators[y][x][num_style_channels];
            if (sumWeight > 0.0001f)
            {
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float val = accumulators[y][x][c] / sumWeight;
                    target[y][x][c] = (uint8_t)std::min(std::max(val, 0.0f), 255.0f);
                }
            }
        }
    }
}

void krnlVotePopulate_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<float, 3> accumulators,
    int patch_size,
    int target_h, int target_w)
{
    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    const int source_h = source.size(0);
    const int source_w = source.size(1);

    // Populate accumulators by splatting all patches
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int ty = 0; ty < target_h; ++ty)
    {
        for (int tx = 0; tx < target_w; ++tx)
        {
            int sx = nnf[ty][tx][0];
            int sy = nnf[ty][tx][1];
            const float error = error_map[ty][tx];
            const float weight = 1.0f / (1.0f + error);

            for (int py = -r; py <= r; py++)
            {
                for (int px = -r; px <= r; px++)
                {
                    int target_x = tx + px;
                    int target_y = ty + py;

                    if (target_x >= 0 && target_x < target_w && target_y >= 0 && target_y < target_h)
                    {
                        int src_x = sx + px;
                        int src_y = sy + py;

                        if (src_x >= 0 && src_x < source_w && src_y >= 0 && src_y < source_h)
                        {
                            for (int c = 0; c < num_style_channels; ++c)
                            {
#pragma omp atomic
                                accumulators[target_y][target_x][c] += weight * (float)source[src_y][src_x][c];
                            }
#pragma omp atomic
                            accumulators[target_y][target_x][num_style_channels] += weight;
                        }
                    }
                }
            }
        }
    }
}
