// ReEzSynth/ebsynth_extension/cpu/patchmatch_cpu.cpp
#include "patchmatch_cpu.h"

#include "cost_functions_cpu.h"
#include "omega_ops_cpu.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <random>

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// Cost function mode constants
#define COST_FUNCTION_SSD 0
#define COST_FUNCTION_NCC 1

// ===================================================================
//                        PATCHMATCH OPERATIONS
// ===================================================================

void try_patch_cpu(
    int candidate_sx, int candidate_sy,
    int tx, int ty, int patch_size,
    torch::PackedTensorAccessor32<int32_t, 3> &nnf,
    torch::PackedTensorAccessor32<float, 2> &error_map,
    torch::PackedTensorAccessor32<int32_t, 2> &omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float uniformity_weight, int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat)
{

    const int source_w = source_style.size(1);
    const int source_h = source_style.size(0);

    if (candidate_sx < patch_size / 2 || candidate_sx >= source_w - patch_size / 2 ||
        candidate_sy < patch_size / 2 || candidate_sy >= source_h - patch_size / 2)
    {
        return;
    }

    float patch_pixel_count = (float)(patch_size * patch_size);
    float omega_best = (float)(nnf.size(0) * nnf.size(1)) / (float)(source_h * source_w) * patch_pixel_count;
    if (omega_best < 1e-6)
        omega_best = 1e-6;

    int current_sx = nnf[ty][tx][0];
    int current_sy = nnf[ty][tx][1];

    float current_ssd = error_map[ty][tx];
    float current_omega_score = patch_omega_cpu(omega_map, current_sx, current_sy, patch_size) / patch_pixel_count / omega_best;
    float current_total_error = current_ssd + uniformity_weight * current_omega_score;

    float new_ssd;
    if (cost_function_mode == COST_FUNCTION_NCC)
    {
        new_ssd = compute_patch_ncc_sat_cpu(source_style, target_style, source_guide, target_guide,
                                            target_modulation_guide, use_modulation, candidate_sx, candidate_sy,
                                            tx, ty, patch_size, style_weights, guide_weights,
                                            source_style_sat, source_style_sq_sat, target_style_sat, target_style_sq_sat);
    }
    else
    {
        new_ssd = compute_patch_ssd_split_cpu(source_style, target_style, source_guide, target_guide,
                                              target_modulation_guide, use_modulation, candidate_sx, candidate_sy,
                                              tx, ty, patch_size, style_weights, guide_weights, current_total_error);
    }

    float new_omega_score = patch_omega_cpu(omega_map, candidate_sx, candidate_sy, patch_size) / patch_pixel_count / omega_best;
    float new_total_error = new_ssd + uniformity_weight * new_omega_score;

    if (new_total_error < current_total_error)
    {
        update_omega_cpu(omega_map, current_sx, current_sy, patch_size, -1);
        update_omega_cpu(omega_map, candidate_sx, candidate_sy, patch_size, 1);
        error_map[ty][tx] = new_ssd;
        nnf[ty][tx][0] = candidate_sx;
        nnf[ty][tx][1] = candidate_sy;
    }
}

void compute_initial_error_cpu(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int cost_function_mode,
    int target_h, int target_w)
{

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int y = 0; y < target_h; ++y)
    {
        for (int x = 0; x < target_w; ++x)
        {
            int sx = nnf[y][x][0];
            int sy = nnf[y][x][1];

            if (cost_function_mode == COST_FUNCTION_NCC)
            {
                error_map[y][x] = compute_patch_ncc_split_cpu(source_style, target_style, source_guide, target_guide,
                                                              target_modulation_guide, use_modulation, sx, sy, x, y,
                                                              patch_size, style_weights, guide_weights,
                                                              std::numeric_limits<float>::max());
            }
            else
            {
                error_map[y][x] = compute_patch_ssd_split_cpu(source_style, target_style, source_guide, target_guide,
                                                              target_modulation_guide, use_modulation, sx, sy, x, y,
                                                              patch_size, style_weights, guide_weights,
                                                              std::numeric_limits<float>::max());
            }
        }
    }
}

void propagation_step_cpu(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, bool is_odd, float uniformity_weight,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    int target_h, int target_w,
    std::vector<ebsynth::PatchCoord> &active_patches)
{
    const int step = is_odd ? -1 : 1;

    if (is_odd)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int i = (int)active_patches.size() - 1; i >= 0; --i)
        {
            const auto &p = active_patches[i];
            int x = p.x;
            int y = p.y;

            if (mask[y][x] == 0)
                continue;

            const int nx1 = x + step;
            if (nx1 >= 0 && nx1 < target_w)
            {
                try_patch_cpu(nnf[y][nx1][0] - step, nnf[y][nx1][1], x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
            }

            const int ny2 = y + step;
            if (ny2 >= 0 && ny2 < target_h)
            {
                try_patch_cpu(nnf[ny2][x][0], nnf[ny2][x][1] - step, x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int i = 0; i < (int)active_patches.size(); ++i)
        {
            const auto &p = active_patches[i];
            int x = p.x;
            int y = p.y;

            if (mask[y][x] == 0)
                continue;

            const int nx1 = x + step;
            if (nx1 >= 0 && nx1 < target_w)
            {
                try_patch_cpu(nnf[y][nx1][0] - step, nnf[y][nx1][1], x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
            }

            const int ny2 = y + step;
            if (ny2 >= 0 && ny2 < target_h)
            {
                try_patch_cpu(nnf[ny2][x][0], nnf[ny2][x][1] - step, x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
            }
        }
    }
}

void random_search_step_cpu(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, int radius, float uniformity_weight,
    std::mt19937 &rng,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    int target_h, int target_w,
    std::vector<ebsynth::PatchCoord> &active_patches)
{

#ifdef _OPENMP
#pragma omp parallel
    {
        std::mt19937 local_rng(rng() + omp_get_thread_num());

#pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < (int)active_patches.size(); ++i)
        {
            const auto &p = active_patches[i];
            int x = p.x;
            int y = p.y;

            if (mask[y][x] == 0)
                continue;

            if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold)
            {
                continue;
            }

            int current_sx = nnf[y][x][0];
            int current_sy = nnf[y][x][1];

            int r = radius;
            while (r >= 1)
            {
                // Generate random offset using thread-local RNG
                unsigned int rand_val1 = static_cast<unsigned int>(local_rng());
                unsigned int rand_val2 = static_cast<unsigned int>(local_rng());
                int candidate_sx = current_sx + static_cast<int>(rand_val1 % (2 * r + 1)) - r;
                int candidate_sy = current_sy + static_cast<int>(rand_val2 % (2 * r + 1)) - r;

                try_patch_cpu(candidate_sx, candidate_sy, x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
                r /= 2;
            }
        }
    }
#else
    for (int i = 0; i < (int)active_patches.size(); ++i)
    {
        const auto &p = active_patches[i];
        int x = p.x;
        int y = p.y;

        if (mask[y][x] == 0)
            continue;

        if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold)
        {
            continue;
        }

        int current_sx = nnf[y][x][0];
        int current_sy = nnf[y][x][1];

        int r = radius;
        while (r >= 1)
        {
            unsigned int rand_val1 = static_cast<unsigned int>(rng());
            unsigned int rand_val2 = static_cast<unsigned int>(rng());
            int candidate_sx = current_sx + static_cast<int>(rand_val1 % (2 * r + 1)) - r;
            int candidate_sy = current_sy + static_cast<int>(rand_val2 % (2 * r + 1)) - r;

            try_patch_cpu(candidate_sx, candidate_sy, x, y, patch_size,
                          nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                          target_modulation_guide, use_modulation, style_weights, guide_weights,
                          uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                          target_style_sat, target_style_sq_sat);
            r /= 2;
        }
    }
#endif
}

void propagation_step_cpu(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, bool is_odd, float uniformity_weight,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    int target_h, int target_w)
{
    const int step = is_odd ? -1 : 1;

    // Full grid iteration (Original baseline)
    if (is_odd)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = target_h - 1; y >= 0; y--)
        {
            for (int x = target_w - 1; x >= 0; x--)
            {
                if (mask[y][x] == 0)
                    continue;

                // Propagate from neighbors (nx+1, ny+1 because step is -1)
                int nx = x + 1;
                if (nx < target_w)
                {
                    try_patch_cpu(nnf[y][nx][0] - 1, nnf[y][nx][1], x, y, patch_size,
                                  nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                                  target_modulation_guide, use_modulation, style_weights, guide_weights,
                                  uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                                  target_style_sat, target_style_sq_sat);
                }

                int ny = y + 1;
                if (ny < target_h)
                {
                    try_patch_cpu(nnf[ny][x][0], nnf[ny][x][1] - 1, x, y, patch_size,
                                  nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                                  target_modulation_guide, use_modulation, style_weights, guide_weights,
                                  uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                                  target_style_sat, target_style_sq_sat);
                }
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < target_h; y++)
        {
            for (int x = 0; x < target_w; x++)
            {
                if (mask[y][x] == 0)
                    continue;

                // Propagate from neighbors (nx-1, ny-1)
                int nx = x - 1;
                if (nx >= 0)
                {
                    try_patch_cpu(nnf[y][nx][0] + 1, nnf[y][nx][1], x, y, patch_size,
                                  nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                                  target_modulation_guide, use_modulation, style_weights, guide_weights,
                                  uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                                  target_style_sat, target_style_sq_sat);
                }

                int ny = y - 1;
                if (ny >= 0)
                {
                    try_patch_cpu(nnf[ny][x][0], nnf[ny][x][1] + 1, x, y, patch_size,
                                  nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                                  target_modulation_guide, use_modulation, style_weights, guide_weights,
                                  uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                                  target_style_sat, target_style_sq_sat);
                }
            }
        }
    }
}

void random_search_step_cpu(
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int patch_size, int radius, float uniformity_weight,
    std::mt19937 &rng,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    int target_h, int target_w)
{
// Full grid random search (Original baseline)
#ifdef _OPENMP
#pragma omp parallel
    {
        std::mt19937 local_rng(rng() + omp_get_thread_num());
#pragma omp for schedule(dynamic, 8)
        for (int y = 0; y < target_h; ++y)
        {
            for (int x = 0; x < target_w; ++x)
            {
                if (mask[y][x] == 0)
                    continue;

                if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold)
                    continue;

                int current_sx = nnf[y][x][0];
                int current_sy = nnf[y][x][1];

                int r = radius;
                while (r >= 1)
                {
                    unsigned int rand_val1 = static_cast<unsigned int>(local_rng());
                    unsigned int rand_val2 = static_cast<unsigned int>(local_rng());
                    int candidate_sx = current_sx + static_cast<int>(rand_val1 % (2 * r + 1)) - r;
                    int candidate_sy = current_sy + static_cast<int>(rand_val2 % (2 * r + 1)) - r;

                    try_patch_cpu(candidate_sx, candidate_sy, x, y, patch_size,
                                  nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                                  target_modulation_guide, use_modulation, style_weights, guide_weights,
                                  uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                                  target_style_sat, target_style_sq_sat);
                    r /= 2;
                }
            }
        }
    }
#else
    for (int y = 0; y < target_h; ++y)
    {
        for (int x = 0; x < target_w; ++x)
        {
            if (mask[y][x] == 0)
                continue;

            if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold)
                continue;

            int current_sx = nnf[y][x][0];
            int current_sy = nnf[y][x][1];

            int r = radius;
            while (r >= 1)
            {
                unsigned int rand_val1 = static_cast<unsigned int>(rng());
                unsigned int rand_val2 = static_cast<unsigned int>(rng());
                int candidate_sx = current_sx + static_cast<int>(rand_val1 % (2 * r + 1)) - r;
                int candidate_sy = current_sy + static_cast<int>(rand_val2 % (2 * r + 1)) - r;

                try_patch_cpu(candidate_sx, candidate_sy, x, y, patch_size,
                              nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide,
                              target_modulation_guide, use_modulation, style_weights, guide_weights,
                              uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat,
                              target_style_sat, target_style_sq_sat);
                r /= 2;
            }
        }
    }
#endif
}
