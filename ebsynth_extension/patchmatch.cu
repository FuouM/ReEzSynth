// ReEzSynth/ebsynth_extension/patchmatch.cu
#include "patchmatch.h"

#include "cost_functions.h"
#include "omega_ops.h"

#include <cmath>
#include <limits>

// ===================================================================
//                        PATCHMATCH KERNELS
// ===================================================================

__device__ void try_patch(
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

  float patch_pixel_count = patch_size * patch_size;
  float omega_best = (float)(nnf.size(0) * nnf.size(1)) / (float)(source_h * source_w) * patch_pixel_count;
  if (omega_best < 1e-6)
    omega_best = 1e-6;

  int current_sx = nnf[ty][tx][0];
  int current_sy = nnf[ty][tx][1];

  float current_ssd = error_map[ty][tx];
  float current_omega_score = patch_omega(omega_map, current_sx, current_sy, patch_size) / patch_pixel_count / omega_best;
  float current_total_error = current_ssd + uniformity_weight * current_omega_score;

  float new_ssd;
  if (cost_function_mode == COST_FUNCTION_NCC)
  {
    new_ssd = compute_patch_ncc_sat(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, candidate_sx, candidate_sy, tx, ty, patch_size, style_weights, guide_weights, source_style_sat, source_style_sq_sat, target_style_sat, target_style_sq_sat);
  }
  else
  {
    new_ssd = compute_patch_ssd_split(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, candidate_sx, candidate_sy, tx, ty, patch_size, style_weights, guide_weights, current_total_error);
  }

  float new_omega_score = patch_omega(omega_map, candidate_sx, candidate_sy, patch_size) / patch_pixel_count / omega_best;
  float new_total_error = new_ssd + uniformity_weight * new_omega_score;

  if (new_total_error < current_total_error)
  {
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
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    int cost_function_mode)
{

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nnf.size(1) || y >= nnf.size(0))
    return;

  int sx = nnf[y][x][0];
  int sy = nnf[y][x][1];
  if (cost_function_mode == COST_FUNCTION_NCC)
  {
    error_map[y][x] = compute_patch_ncc_split(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, sx, sy, x, y, patch_size, style_weights, guide_weights, std::numeric_limits<float>::max());
  }
  else
  {
    error_map[y][x] = compute_patch_ssd_split(source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, sx, sy, x, y, patch_size, style_weights, guide_weights, std::numeric_limits<float>::max());
  }
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
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat)
{

  const int y_raw = blockIdx.y * blockDim.y + threadIdx.y;
  const int x_raw = blockIdx.x * blockDim.x + threadIdx.x;
  const int target_h = nnf.size(0);
  const int target_w = nnf.size(1);

  if (x_raw >= target_w || y_raw >= target_h)
    return;

  const int y = is_odd ? y_raw : (target_h - 1 - y_raw);
  const int x = is_odd ? x_raw : (target_w - 1 - x_raw);

  if (mask[y][x] == 0)
    return;

  const int step = is_odd ? -1 : 1;

  const int nx1 = x + step;
  if (nx1 >= 0 && nx1 < target_w)
  {
    try_patch(nnf[y][nx1][0] - step, nnf[y][nx1][1], x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat, target_style_sat, target_style_sq_sat);
  }

  const int ny2 = y + step;
  if (ny2 >= 0 && ny2 < target_h)
  {
    try_patch(nnf[ny2][x][0], nnf[ny2][x][1] - step, x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat, target_style_sat, target_style_sq_sat);
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
    int patch_size, int radius, float uniformity_weight, curandState *states,
    torch::PackedTensorAccessor32<uint8_t, 2> mask,
    float search_pruning_threshold,
    int cost_function_mode,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat)
{

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nnf.size(1) || y >= nnf.size(0))
    return;
  if (mask[y][x] == 0)
    return;

  if (search_pruning_threshold > 0.0f && error_map[y][x] < search_pruning_threshold)
  {
    return;
  }

  int idx = y * nnf.size(1) + x;
  curandState *state = &states[idx];

  int current_sx = nnf[y][x][0];
  int current_sy = nnf[y][x][1];

  int r = radius;
  while (r >= 1)
  {
    int candidate_sx = current_sx + (curand(state) % (2 * r + 1)) - r;
    int candidate_sy = current_sy + (curand(state) % (2 * r + 1)) - r;

    try_patch(candidate_sx, candidate_sy, x, y, patch_size, nnf, error_map, omega_map, source_style, target_style, source_guide, target_guide, target_modulation_guide, use_modulation, style_weights, guide_weights, uniformity_weight, cost_function_mode, source_style_sat, source_style_sq_sat, target_style_sat, target_style_sq_sat);
    r /= 2;
  }
}
