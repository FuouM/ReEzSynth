// ReEzSynth/ebsynth_extension/integral_image.cu
#include "integral_image.h"

// Each block processes a single row
__global__ void prepare_and_row_prefix_sum_kernel(
    double* intermediate,
    torch::PackedTensorAccessor32<uint8_t, 3> input,
    int width, int height,
    PrepMode mode)
{
    const int y = blockIdx.x;
    if (y >= height) return;

    double sum = 0.0;
    for (int x = 0; x < width; ++x) {
        // Prepare: convert to grayscale double
        double val = 0.0;
        const int num_channels = input.size(2);
        for(int c=0; c < num_channels; ++c) {
            val += (double)input[y][x][c];
        }
        val /= num_channels;

        if (mode == PREP_GRAY_SQR) {
            val = val * val;
        }

        sum += val;
        intermediate[y * width + x] = sum;
    }
}

// Each block processes a single column
__global__ void col_prefix_sum_kernel(
    torch::PackedTensorAccessor64<double, 2> output_sat,
    const double* intermediate,
    int width, int height)
{
    const int x = blockIdx.x;
    if (x >= width) return;

    double sum = 0.0;
    for (int y = 0; y < height; ++y) {
        sum += intermediate[y * width + x];
        output_sat[y][x] = sum;
    }
}

void compute_integral_image_cuda(
    torch::Tensor output_sat,
    torch::Tensor input_image,
    PrepMode mode)
{
    const int height = input_image.size(0);
    const int width = input_image.size(1);

    auto options = torch::TensorOptions().device(input_image.device()).dtype(torch::kFloat64);
    torch::Tensor intermediate = torch::empty({height * width}, options);

    // Pass 1: Row-wise prefix sum
    dim3 blocks_rows(height);
    dim3 threads_rows(1); // Simplistic launch, could be improved with block-wide scans
    prepare_and_row_prefix_sum_kernel<<<blocks_rows, threads_rows>>>(
        intermediate.data_ptr<double>(),
        input_image.packed_accessor32<uint8_t, 3>(),
        width, height,
        mode
    );

    // Pass 2: Column-wise prefix sum
    dim3 blocks_cols(width);
    dim3 threads_cols(1);
    col_prefix_sum_kernel<<<blocks_cols, threads_cols>>>(
        output_sat.packed_accessor64<double, 2>(),
        intermediate.data_ptr<double>(),
        width, height
    );
}