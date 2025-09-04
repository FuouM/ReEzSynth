// ReEzSynth/ebsynth_extension/kernels.cu
#include "kernels.h"

#include <iostream>
#include <limits>
#include <vector>

// ===================================================================
//                        UNIFORMITY KERNELS
// ===================================================================

__device__ void update_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size, int incdec) {
    const int r = patch_size / 2;
    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_map.size(1) && cur_sy >= 0 && cur_sy < omega_map.size(0)) {
                atomicAdd(&omega_map[cur_sy][cur_sx], incdec);
            }
        }
    }
}

__device__ float patch_omega(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    int sx, int sy, int patch_size) {
    const int r = patch_size / 2;
    float sum = 0;
    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            if (cur_sx >= 0 && cur_sx < omega_map.size(1) && cur_sy >= 0 && cur_sy < omega_map.size(0)) {
                sum += omega_map[cur_sy][cur_sx];
            }
        }
    }
    return sum;
}

__global__ void populate_initial_omega_kernel(
    torch::PackedTensorAccessor32<int32_t, 2> omega_map,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= nnf.size(1) || ty >= nnf.size(0)) return;

    int sx = nnf[ty][tx][0];
    int sy = nnf[ty][tx][1];
    update_omega(omega_map, sx, sy, patch_size, 1);
}

// ===================================================================
//                        MASKING KERNELS
// ===================================================================

__global__ void eval_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    torch::PackedTensorAccessor32<uint8_t, 3> style1,
    torch::PackedTensorAccessor32<uint8_t, 3> style2,
    int stop_threshold) {
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mask.size(1) || y >= mask.size(0)) return;

    const int num_channels = style1.size(2);
    int max_diff = 0;
    for (int c = 0; c < num_channels; ++c) {
        int diff = abs((int)style1[y][x][c] - (int)style2[y][x][c]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    mask[y][x] = (max_diff < stop_threshold) ? 0 : 255;
}

__global__ void dilate_mask_kernel(
    torch::PackedTensorAccessor32<uint8_t, 2> mask_out,
    torch::PackedTensorAccessor32<uint8_t, 2> mask_in,
    int patch_size) {
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mask_out.size(1) || y >= mask_out.size(0)) return;

    const int r = patch_size / 2;
    uint8_t msk_val = 0;

    for (int py = -r; py <= r; ++py) {
        for (int px = -r; px <= r; ++px) {
            int nx = x + px;
            int ny = y + py;
            if (nx >= 0 && nx < mask_in.size(1) && ny >= 0 && ny < mask_in.size(0)) {
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


// ===================================================================
//                        PATCHMATCH KERNELS
// ===================================================================

__device__ float compute_patch_ssd_split(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest) {
  
  const int r = patch_size / 2;
  float error = 0.0f;

  const int num_style_channels = source_style.size(2);
  const int num_guide_channels = source_guide.size(2);

  const int source_h = source_style.size(0);
  const int source_w = source_style.size(1);
  const int target_h = target_style.size(0);
  const int target_w = target_style.size(1);

  for (int py = -r; py <= r; ++py) {
    for (int px = -r; px <= r; ++px) {
      int cur_sx = min(max(sx + px, 0), source_w - 1);
      int cur_sy = min(max(sy + py, 0), source_h - 1);
      int cur_tx = min(max(tx + px, 0), target_w - 1);
      int cur_ty = min(max(ty + py, 0), target_h - 1);

      // Style difference
      for (int c = 0; c < num_style_channels; ++c) {
        float diff = (float)source_style[cur_sy][cur_sx][c] - (float)target_style[cur_ty][cur_tx][c];
        error += style_weights[c] * diff * diff;
      }

      // Guide difference
      for (int c = 0; c < num_guide_channels; ++c) {
        float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
        float modulation = 1.0f;
        if (use_modulation) {
            modulation = (float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f;
        }
        error += guide_weights[c] * modulation * diff * diff;
      }
    }
    if(error > ebest) return error;
  }
  return error;
}

__device__ void try_patch(
    int candidate_sx, int candidate_sy,
    int tx, int ty, int patch_size,
    torch::PackedTensorAccessor32<int32_t, 3>& nnf,
    torch::PackedTensorAccessor32<float, 2>& error_map,
    torch::PackedTensorAccessor32<int32_t, 2>& omega_map,
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float uniformity_weight) {

    const int source_w = source_style.size(1);
    const int source_h = source_style.size(0);
    if (candidate_sx < patch_size/2 || candidate_sx >= source_w - patch_size/2 ||
        candidate_sy < patch_size/2 || candidate_sy >= source_h - patch_size/2) {
        return;
    }

    float patch_pixel_count = patch_size * patch_size;
    float omega_best = (float)(nnf.size(0) * nnf.size(1)) / (float)(source_h * source_w) * patch_pixel_count;
    if (omega_best < 1e-6) omega_best = 1e-6;

    int current_sx = nnf[ty][tx][0];
    int current_sy = nnf[ty][tx][1];

    float current_ssd = error_map[ty][tx];
    float current_omega_score = patch_omega(omega_map, current_sx, current_sy, patch_size) / patch_pixel_count / omega_best;
    float current_total_error = current_ssd + uniformity_weight * current_omega_score;

    float new_ssd = compute_patch_ssd_split(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, candidate_sx, candidate_sy, tx, ty, patch_size, style_weights, guide_weights, current_total_error);
    float new_omega_score = patch_omega(omega_map, candidate_sx, candidate_sy, patch_size) / patch_pixel_count / omega_best;
    float new_total_error = new_ssd + uniformity_weight * new_omega_score;

    if (new_total_error < current_total_error) {
        update_omega(omega_map, current_sx, current_sy, patch_size, -1);
        update_omega(omega_map, candidate_sx, candidate_sy, patch_size, 1);
        error_map[ty][tx] = new_ssd;
        nnf[ty][tx][0] = candidate_sx;
        nnf[ty][tx][1] = candidate_sy;
    }
}


__global__ void compute_initial_error_kernel(
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
    const torch::PackedTensorAccessor32<float, 1> guide_weights) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nnf.size(1) || y >= nnf.size(0)) return;

  int sx = nnf[y][x][0];
  int sy = nnf[y][x][1];
  error_map[y][x] = compute_patch_ssd_split(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, sx, sy, x, y, patch_size, style_weights, guide_weights, std::numeric_limits<float>::max());
}

__global__ void propagation_step_kernel(
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
    torch::PackedTensorAccessor32<uint8_t, 2> mask) {

  const int y_raw = blockIdx.y * blockDim.y + threadIdx.y;
  const int x_raw = blockIdx.x * blockDim.x + threadIdx.x;
  const int target_h = nnf.size(0);
  const int target_w = nnf.size(1);

  if (x_raw >= target_w || y_raw >= target_h) return;

  const int y = is_odd ? y_raw : (target_h - 1 - y_raw);
  const int x = is_odd ? x_raw : (target_w - 1 - x_raw);
  
  if (mask[y][x] == 0) return;

  const int step = is_odd ? -1 : 1;
  
  const int nx1 = x + step;
  if (nx1 >= 0 && nx1 < target_w) {
    try_patch(nnf[y][nx1][0] - step, nnf[y][nx1][1], x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight);
  }

  const int ny2 = y + step;
  if (ny2 >= 0 && ny2 < target_h) {
    try_patch(nnf[ny2][x][0], nnf[ny2][x][1] - step, x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight);
  }
}

__global__ void random_search_step_kernel(
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
    int patch_size, int radius, float uniformity_weight, curandState* states,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold) {
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nnf.size(1) || y >= nnf.size(0)) return;
    if (mask[y][x] == 0) return;

    if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold) {
        return;
    }

    int idx = y * nnf.size(1) + x;
    curandState* state = &states[idx];

    int current_sx = nnf[y][x][0];
    int current_sy = nnf[y][x][1];
    
    int r = radius;
    while (r >= 1) {
        int candidate_sx = current_sx + (curand(state) % (2 * r + 1)) - r;
        int candidate_sy = current_sy + (curand(state) % (2 * r + 1)) - r;

        try_patch(candidate_sx, candidate_sy, x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight);
        r /= 2;
    }
}

// ===================================================================
//                        VOTING KERNELS
// ===================================================================

__global__ void krnlVotePlain(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= target.size(1) || y >= target.size(0)) return;

    const int r = patch_size / 2;
    const int num_style_channels = source.size(2);
    Vec<4, float> sumColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float sumWeight = 0.0f;

    for (int py = -r; py <= r; py++) {
        for (int px = -r; px <= r; px++) {
            int t_neighbor_x = x - px;
            int t_neighbor_y = y - py;
            if (t_neighbor_x >= 0 && t_neighbor_x < target.size(1) && t_neighbor_y >= 0 && t_neighbor_y < target.size(0)) {
                int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
                int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

                int source_x = s_neighbor_center_x + px;
                int source_y = s_neighbor_center_y + py;

                if (source_x >= 0 && source_x < source.size(1) && source_y >= 0 && source_y < source.size(0)) {
                    const float weight = 1.0f;
                    for (int c = 0; c < num_style_channels; ++c) {
                        sumColor.v[c] += weight * (float)source[source_y][source_x][c];
                    }
                    sumWeight += weight;
                }
            }
        }
    }
  
    if (sumWeight > 0.0001f) {
        for (int c = 0; c < num_style_channels; ++c) {
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
    int patch_size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= target.size(1) || y >= target.size(0)) return;

  const int r = patch_size / 2;
  const int num_style_channels = source.size(2);
  Vec<4, float> sumColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
  float sumWeight = 0.0f;

  for (int py = -r; py <= r; py++) {
    for (int px = -r; px <= r; px++) {
      int t_neighbor_x = x - px;
      int t_neighbor_y = y - py;
      
      if (t_neighbor_x >= 0 && t_neighbor_x < target.size(1) &&
          t_neighbor_y >= 0 && t_neighbor_y < target.size(0)) {
        
        int s_neighbor_center_x = nnf[t_neighbor_y][t_neighbor_x][0];
        int s_neighbor_center_y = nnf[t_neighbor_y][t_neighbor_x][1];

        int source_x = s_neighbor_center_x + px;
        int source_y = s_neighbor_center_y + py;
        
        if (source_x >= 0 && source_x < source.size(1) &&
            source_y >= 0 && source_y < source.size(0)) {
              
          const float error = error_map[t_neighbor_y][t_neighbor_x];
          const float weight = 1.0f / (1.0f + error);
          
          for (int c = 0; c < num_style_channels; ++c) {
            sumColor.v[c] += weight * (float)source[source_y][source_x][c];
          }
          sumWeight += weight;
        }
      }
    }
  }
  
  if (sumWeight > 0.0001f) {
    for (int c = 0; c < num_style_channels; ++c) {
      float val = sumColor.v[c] / sumWeight;
      target[y][x][c] = (uint8_t)fminf(fmaxf(val, 0.0f), 255.0f);
    }
  }
}

// ===================================================================
//                RNG STATE INITIALIZER KERNEL
// ===================================================================
__global__ void init_rand_states_kernel(curandState* states, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(1337, idx, 0, &states[idx]);
    }
}