// ReEzSynth/ezsynth/ebsynth_extension/integral_image.cu
#include "integral_image.h"

namespace
{
constexpr int kScanThreads = 128;
}

// One thread per row: converts the row to grayscale and writes its prefix sum.
// Each row is still summed serially left-to-right, so results match the CPU path.
__global__ void prepare_and_row_prefix_sum_kernel(
    double *intermediate,
    ebs::PackedAccessor32<uint8_t, 3> input,
    int width, int height,
    PrepMode mode)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height)
        return;

    const int num_channels = input.size(2);
    double sum = 0.0;
    for (int x = 0; x < width; ++x)
    {
        double val = 0.0;
        for (int c = 0; c < num_channels; ++c)
        {
            val += (double)input[y][x][c];
        }
        val /= num_channels;

        if (mode == PREP_GRAY_SQR)
        {
            val = val * val;
        }

        sum += val;
        intermediate[y * width + x] = sum;
    }
}

// One thread per column; adjacent threads touch adjacent columns (coalesced).
__global__ void col_prefix_sum_kernel(
    ebs::PackedAccessor64<double, 2> output_sat,
    const double *intermediate,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
        return;

    double sum = 0.0;
    for (int y = 0; y < height; ++y)
    {
        sum += intermediate[y * width + x];
        output_sat[y][x] = sum;
    }
}

cudaError_t compute_integral_image_cuda(
    ebs::PackedAccessor64<double, 2> output_sat,
    ebs::PackedAccessor32<uint8_t, 3> input_image,
    double *intermediate,
    PrepMode mode,
    cudaStream_t stream)
{
    const int height = input_image.size(0);
    const int width = input_image.size(1);

    // Pass 1: Row-wise prefix sum
    prepare_and_row_prefix_sum_kernel<<<(height + kScanThreads - 1) / kScanThreads, kScanThreads, 0, stream>>>(
        intermediate, input_image, width, height, mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Pass 2: Column-wise prefix sum
    col_prefix_sum_kernel<<<(width + kScanThreads - 1) / kScanThreads, kScanThreads, 0, stream>>>(
        output_sat, intermediate, width, height);
    return cudaGetLastError();
}
