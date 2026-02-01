// ReEzSynth/ebsynth_extension/cpu/mask_ops_cpu.cpp
#include "mask_ops_cpu.h"

#include <cstdlib>
#include <algorithm>

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// ===================================================================
//                        MASKING OPERATIONS
// ===================================================================

void eval_mask_cpu(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold,
    int target_h, int target_w) {

    const int num_channels = style1.size(2);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
    #endif
    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            int max_diff = 0;
            for (int c = 0; c < num_channels; ++c) {
                int diff = std::abs((int)style1[y][x][c] - (int)style2[y][x][c]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
            mask[y][x] = (max_diff < stop_threshold) ? 0 : 255;
        }
    }
}

void dilate_mask_cpu(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size,
    int target_h, int target_w) {

    const int r = patch_size / 2;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
    #endif
    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            uint8_t msk_val = 0;

            for (int py = -r; py <= r; ++py) {
                for (int px = -r; px <= r; ++px) {
                    int nx = x + px;
                    int ny = y + py;
                    if (nx >= 0 && nx < target_w && ny >= 0 && ny < target_h) {
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
    }
}
