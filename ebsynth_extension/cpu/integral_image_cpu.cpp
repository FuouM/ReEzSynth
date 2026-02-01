// ReEzSynth/ebsynth_extension/cpu/integral_image_cpu.cpp
#include "integral_image_cpu.h"

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// Computes a summed-area table (integral image) from a uint8 image.
void compute_integral_image_cpu(
    torch::Tensor output_sat,
    torch::Tensor input_image,
    PrepMode mode) {

    const int height = input_image.size(0);
    const int width = input_image.size(1);
    const int num_channels = input_image.size(2);

    auto input_acc = input_image.packed_accessor32<uint8_t, 3>();
    auto output_acc = output_sat.packed_accessor64<double, 2>();

    // First pass: compute row-wise prefix sums in parallel
    // Each row is independent
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < height; ++y) {
        double row_sum = 0.0;
        for (int x = 0; x < width; ++x) {
            // Convert to grayscale
            double val = 0.0;
            for (int c = 0; c < num_channels; ++c) {
                val += (double)input_acc[y][x][c];
            }
            val /= num_channels;

            if (mode == PREP_GRAY_SQR) {
                val = val * val;
            }

            row_sum += val;
            output_acc[y][x] = row_sum;
        }
    }

    // Second pass: column-wise prefix sums
    // This has a dependency on the previous row, so we process columns in parallel
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int x = 0; x < width; ++x) {
        double col_sum = 0.0;
        for (int y = 0; y < height; ++y) {
            col_sum += output_acc[y][x];
            output_acc[y][x] = col_sum;
        }
    }
}
