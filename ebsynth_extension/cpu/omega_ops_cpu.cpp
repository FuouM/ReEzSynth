// ReEzSynth/ebsynth_extension/cpu/omega_ops_cpu.cpp
#include "omega_ops_cpu.h"

#include <algorithm>

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// ===================================================================
//                        UNIFORMITY OPERATIONS
// ===================================================================

void update_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec) {

    const int r = patch_size / 2;
    const int omega_h = omega_map.size(0);
    const int omega_w = omega_map.size(1);

    // Simple nested loop - not worth parallelizing for small patches
    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_w && cur_sy >= 0 && cur_sy < omega_h) {
                omega_map[cur_sy][cur_sx] += incdec;
            }
        }
    }
}

float patch_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size) {

    const int r = patch_size / 2;
    const int omega_h = omega_map.size(0);
    const int omega_w = omega_map.size(1);

    float sum = 0;
    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_w && cur_sy >= 0 && cur_sy < omega_h) {
                sum += omega_map[cur_sy][cur_sx];
            }
        }
    }
    return sum;
}

void populate_initial_omega_cpu(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size,
    int target_h, int target_w) {

    // Initialize omega map to zeros
    const int omega_h = omega_map.size(0);
    const int omega_w = omega_map.size(1);

    // Parallelize initialization
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < omega_h; ++y) {
        for (int x = 0; x < omega_w; ++x) {
            omega_map[y][x] = 0;
        }
    }

    // Populate from NNF - this has write contention, so we use atomic updates
    // or process in a way that avoids conflicts
    // For now, use a simple approach with critical sections for updates
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
    #endif
    for (int ty = 0; ty < target_h; ++ty) {
        for (int tx = 0; tx < target_w; ++tx) {
            int sx = nnf[ty][tx][0];
            int sy = nnf[ty][tx][1];

            const int r = patch_size / 2;

            // Update omega map with atomic operations to avoid race conditions
            for (int py = -r; py <= r; ++py) {
                for (int px = -r; px <= r; ++px) {
                    int cur_sx = sx + px;
                    int cur_sy = sy + py;
                    if (cur_sx >= 0 && cur_sx < omega_w && cur_sy >= 0 && cur_sy < omega_h) {
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        omega_map[cur_sy][cur_sx] += 1;
                    }
                }
            }
        }
    }
}
