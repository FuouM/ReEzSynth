// FastBlend CUDA Kernels

#include "kernels.h"

#include <iostream>
#include <limits>
#include <vector>

// Remapping kernel - applies NNF to remap source style to target
__global__ void remap_kernel(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* __restrict__ source_style,
    const int* __restrict__ nnf,
    float* __restrict__ target_style
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= height || y >= width) return;

    const int batch_idx = blockIdx.z;

    const int z = batch_idx * (height + pad_size * 2) * (width + pad_size * 2) * channel;
    const int pid = (x + pad_size) * (width + pad_size * 2) + (y + pad_size);
    const int min_px = x < r ? -x : -r;
    const int max_px = x + r > height - 1 ? height - 1 - x : r;
    const int min_py = y < r ? -y : -r;
    const int max_py = y + r > width - 1 ? width - 1 - y : r;

    // Initialize target to zero
    for (int c = 0; c < channel; c++) {
        target_style[z + pid * channel + c] = 0.0f;
    }

    int num = 0;
    // Accumulate patch values
    for (int px = min_px; px <= max_px; px++) {
        for (int py = min_py; py <= max_py; py++) {
            const int nid = (x + px) * width + y + py;
            const int x_ = nnf[batch_idx * height * width * 2 + nid * 2 + 0] - px;  // NNF[0] = x
            const int y_ = nnf[batch_idx * height * width * 2 + nid * 2 + 1] - py;  // NNF[1] = y
            if (x_ < 0 || y_ < 0 || x_ >= height || y_ >= width) continue;
            const int pid_ = (x_ + pad_size) * (width + pad_size * 2) + (y_ + pad_size);
            num++;
            for (int c = 0; c < channel; c++) {
                target_style[z + pid * channel + c] += source_style[z + pid_ * channel + c];
            }
        }
    }

    // Divide by number of samples
    if (num > 0) {
        for (int c = 0; c < channel; c++) {
            target_style[z + pid * channel + c] /= num;
        }
    }
}

// Patch error kernel - computes patch matching error
__global__ void patch_error_kernel(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* __restrict__ source,
    const int* __restrict__ nnf,
    const float* __restrict__ target,
    float* __restrict__ error
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    const int z = batch_idx * (height + pad_size * 2) * (width + pad_size * 2) * channel;

    if (x >= height || y >= width) return;

    const int nnf_idx = batch_idx * height * width * 2 + (x * width + y) * 2;
    const int x_ = nnf[nnf_idx + 0];  // NNF[0] = x
    const int y_ = nnf[nnf_idx + 1];  // NNF[1] = y

    float e = 0.0f;

    // Unroll loops for small patch sizes and separate RGB channels
    #pragma unroll 3
    for (int px = -r; px <= r; px++) {
        #pragma unroll 3
        for (int py = -r; py <= r; py++) {
            const int pid = (x + pad_size + px) * (width + pad_size * 2) + y + pad_size + py;
            const int pid_ = (x_ + pad_size + px) * (width + pad_size * 2) + y_ + pad_size + py;
            const int idx = z + pid * channel;
            const int idx_ = z + pid_ * channel;

            const float dr = target[idx + 0] - source[idx_ + 0];
            const float dg = target[idx + 1] - source[idx_ + 1];
            const float db = target[idx + 2] - source[idx_ + 2];

            e += dr * dr + dg * dg + db * db;
        }
    }
    error[batch_idx * height * width + x * width + y] = e;
}

// Pairwise patch error kernel - for comparing two different NNFs
__global__ void pairwise_patch_error_kernel(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* __restrict__ source_a,
    const int* __restrict__ nnf_a,
    const float* __restrict__ source_b,
    const int* __restrict__ nnf_b,
    float* __restrict__ error
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    const int z = batch_idx * (height + pad_size * 2) * (width + pad_size * 2) * channel;

    if (x >= height || y >= width) return;

    const int z_nnf = batch_idx * height * width * 2 + (x * width + y) * 2;
    const int x_a = nnf_a[z_nnf + 0];  // NNF[0] = x
    const int y_a = nnf_a[z_nnf + 1];  // NNF[1] = y
    const int x_b = nnf_b[z_nnf + 0];  // NNF[0] = x
    const int y_b = nnf_b[z_nnf + 1];  // NNF[1] = y

    float e = 0.0f;
    for (int px = -r; px <= r; px++) {
        for (int py = -r; py <= r; py++) {
            const int pid_a = (x_a + pad_size + px) * (width + pad_size * 2) + y_a + pad_size + py;
            const int pid_b = (x_b + pad_size + px) * (width + pad_size * 2) + y_b + pad_size + py;
            #pragma unroll
            for (int c = 0; c < channel; c++) {
                const float diff = source_a[z + pid_a * channel + c] - source_b[z + pid_b * channel + c];
                e += diff * diff;
            }
        }
    }
    error[batch_idx * height * width + x * width + y] = e;
}

// Optimized remapping with shared memory for better performance
__global__ void remap_kernel_shared(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* __restrict__ source_style,
    const int* __restrict__ nnf,
    float* __restrict__ target_style,
    const int batch_idx
) {
    extern __shared__ float shared_patch[];

    const int r = (patch_size - 1) / 2;
    const int patch_area = (2 * r + 1) * (2 * r + 1);
    const int shared_patch_size = patch_area * channel;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (x >= height || y >= width) return;

    // Load NNF value for this pixel
    const int nid = x * width + y;
    const int x_ = nnf[batch_idx * height * width * 2 + nid * 2 + 0];  // NNF[0] = x
    const int y_ = nnf[batch_idx * height * width * 2 + nid * 2 + 1];  // NNF[1] = y

    // Clamp bounds
    const int min_px = x < r ? -x : -r;
    const int max_px = x + r > height - 1 ? height - 1 - x : r;
    const int min_py = y < r ? -y : -r;
    const int max_py = y + r > width - 1 ? width - 1 - y : r;

    // Initialize shared memory to zero
    for (int i = tid; i < shared_patch_size; i += blockDim.x * blockDim.y) {
        shared_patch[i] = 0.0f;
    }
    __syncthreads();

    // Accumulate patch values in shared memory
    int count = 0;
    for (int px = min_px; px <= max_px; px++) {
        for (int py = min_py; py <= max_py; py++) {
            const int src_x = x_ - px;
            const int src_y = y_ - py;
            if (src_x >= 0 && src_x < height && src_y >= 0 && src_y < width) {
                const int pid_ = (src_x + pad_size) * (width + pad_size * 2) + (src_y + pad_size);
                const int z = batch_idx * (height + pad_size * 2) * (width + pad_size * 2) * channel;

                #pragma unroll
                for (int c = 0; c < channel; c++) {
                    const int patch_idx = (count * channel) + c;
                    if (patch_idx < shared_patch_size) {
                        shared_patch[patch_idx] += source_style[z + pid_ * channel + c];
                    }
                }
                count++;
            }
        }
    }
    __syncthreads();

    // Write averaged result
    if (count > 0) {
        const int z = batch_idx * (height + pad_size * 2) * (width + pad_size * 2) * channel;
        const int pid = (x + pad_size) * (width + pad_size * 2) + (y + pad_size);

        #pragma unroll
        for (int c = 0; c < channel; c++) {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) {
                const int patch_idx = (i * channel) + c;
                if (patch_idx < shared_patch_size) {
                    sum += shared_patch[patch_idx];
                }
            }
            target_style[z + pid * channel + c] = sum / count;
        }
    }
}

// CUDA kernel launchers
void launch_remap_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_style, const int* nnf, float* target_style,
    const int batch_size
) {
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y, batch_size);

    remap_kernel<<<grid, block>>>(
        height, width, channel, patch_size, pad_size,
        source_style, nnf, target_style
    );
}

void launch_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source, const int* nnf, const float* target, float* error,
    const int batch_size
) {
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y, batch_size);

    patch_error_kernel<<<grid, block>>>(
        height, width, channel, patch_size, pad_size,
        source, nnf, target, error
    );
}

void launch_pairwise_patch_error_kernel(
    const int height, const int width, const int channel, const int patch_size, const int pad_size,
    const float* source_a, const int* nnf_a, const float* source_b, const int* nnf_b, float* error,
    const int batch_size
) {
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y, batch_size);

    pairwise_patch_error_kernel<<<grid, block>>>(
        height, width, channel, patch_size, pad_size,
        source_a, nnf_a, source_b, nnf_b, error
    );
}