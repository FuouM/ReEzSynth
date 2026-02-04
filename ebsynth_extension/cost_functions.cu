// ReEzSynth/ebsynth_extension/cost_functions.cu
#include "cost_functions.h"

#include <cmath>
#include <limits>

// ===================================================================
//                        SSD COST FUNCTION
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
    float ebest,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step)
{

    const int r = patch_size / 2;
    float error = 0.0f;

    const int num_style_channels = source_style.size(2);
    const int num_guide_channels = source_guide.size(2);

    const int source_h = source_style.size(0);
    const int source_w = source_style.size(1);
    const int target_h = target_style.size(0);
    const int target_w = target_style.size(1);

    float weight_sum = 0.0f;

    for (int py = -r; py <= r; py += n_size_step)
    {
        for (int px = -r; px <= r; px += n_size_step)
        {
            int cur_sx = min(max(sx + px, 0), source_w - 1);
            int cur_sy = min(max(sy + py, 0), source_h - 1);
            int cur_tx = min(max(tx + px, 0), target_w - 1);
            int cur_ty = min(max(ty + py, 0), target_h - 1);

            float weight = 1.0f;
            if (use_bilateral)
            {
                float spatial_dist_sq = (float)(px * px + py * py);
                float color_dist_sq = 0.0f;
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float d = (float)source_style[cur_sy][cur_sx][c] - (float)source_style[sy][sx][c];
                    color_dist_sq += d * d;
                }
                weight = expf(-spatial_dist_sq / (2.0f * sigma_spatial * sigma_spatial) - color_dist_sq / (2.0f * sigma_color * sigma_color));
            }
            weight_sum += weight;

            // Style difference
            for (int c = 0; c < num_style_channels; ++c)
            {
                float diff = (float)source_style[cur_sy][cur_sx][c] - (float)target_style[cur_ty][cur_tx][c];
                error += weight * style_weights[c] * diff * diff;
            }

            // Guide difference
            for (int c = 0; c < num_guide_channels; ++c)
            {
                float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
                float modulation = 1.0f;
                if (use_modulation)
                {
                    modulation = (float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f;
                }
                error += weight * guide_weights[c] * modulation * diff * diff;
            }
        }
        if (ebest > 0 && error > ebest)
            return error;
    }
    return (weight_sum > 0) ? (error / weight_sum * (patch_size * patch_size)) : error;
}

// ===================================================================
//                        NCC COST FUNCTIONS
// ===================================================================

__device__ double query_sat(
    torch::PackedTensorAccessor64<double, 2> sat,
    int x1, int y1, int x2, int y2)
{

    const int h = sat.size(0);
    const int w = sat.size(1);

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    x2 = min(x2, w - 1);
    y2 = min(y2, h - 1);

    double br = sat[y2][x2];
    double bl = (x1 > 0) ? sat[y2][x1 - 1] : 0.0;
    double tr = (y1 > 0) ? sat[y1 - 1][x2] : 0.0;
    double tl = (x1 > 0 && y1 > 0) ? sat[y1 - 1][x1 - 1] : 0.0;

    return br - bl - tr + tl;
}

/*
 * NCC cost function using Summed-Area Tables (SATs).
 * This computes patch means and variances in O(1) time.
 * The cross-correlation term still requires an O(P^2) loop.
 */
__device__ float compute_patch_ncc_sat(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    torch::PackedTensorAccessor64<double, 2> source_style_sat,
    torch::PackedTensorAccessor64<double, 2> source_style_sq_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sat,
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step)
{

    const int r = patch_size / 2;
    const float N = patch_size * patch_size;
    const float epsilon = 1e-6f;

    const int num_style_channels = source_style.size(2);
    const int num_guide_channels = source_guide.size(2);

    // --- O(1) Style Stats using SATs ---
    double sum_s = query_sat(source_style_sat, sx - r, sy - r, sx + r, sy + r);
    double sum_sq_s = query_sat(source_style_sq_sat, sx - r, sy - r, sx + r, sy + r);
    double sum_t = query_sat(target_style_sat, tx - r, ty - r, tx + r, ty + r);
    double sum_sq_t = query_sat(target_style_sq_sat, tx - r, ty - r, tx + r, ty + r);

    double mean_s = sum_s / N;
    double mean_t = sum_t / N;
    double std_s = sqrt(fmax(0.0, sum_sq_s / N - mean_s * mean_s));
    double std_t = sqrt(fmax(0.0, sum_sq_t / N - mean_t * mean_t));

    // --- O(P^2) Cross-correlation and Guide SSD ---
    double sum_st = 0.0;
    float guide_error = 0.0f;
    float weight_sum = 0.0f;
    for (int py = -r; py <= r; py += n_size_step)
    {
        for (int px = -r; px <= r; px += n_size_step)
        {
            int cur_sx = sx + px;
            int cur_sy = sy + py;
            int cur_tx = tx + px;
            int cur_ty = ty + py;

            // Cross-correlation term
            float weight = 1.0f;
            if (use_bilateral)
            {
                float spatial_dist_sq = (float)(px * px + py * py);
                float color_dist_sq = 0.0f;
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float d = (float)source_style[cur_sy][cur_sx][c] - (float)source_style[sy][sx][c];
                    color_dist_sq += d * d;
                }
                weight = expf(-spatial_dist_sq / (2.0f * sigma_spatial * sigma_spatial) - color_dist_sq / (2.0f * sigma_color * sigma_color));
            }
            weight_sum += weight;

            float s_val_g = 0.0f, t_val_g = 0.0f;
            for (int c = 0; c < num_style_channels; ++c)
            {
                s_val_g += (float)source_style[cur_sy][cur_sx][c];
                t_val_g += (float)target_style[cur_ty][cur_tx][c];
            }
            sum_st += weight * (s_val_g / num_style_channels) * (t_val_g / num_style_channels);

            // Guide difference (SSD)
            for (int c = 0; c < num_guide_channels; ++c)
            {
                float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
                float modulation = use_modulation ? ((float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f) : 1.0f;
                guide_error += weight * guide_weights[c] * modulation * diff * diff;
            }
        }
    }

    double cov = (weight_sum > 0) ? (sum_st / weight_sum - mean_s * mean_t) : 0.0;
    float ncc = (std_s > epsilon && std_t > epsilon) ? cov / (std_s * std_t) : 0.0f;
    float style_error = (1.0f - ncc) * style_weights[0] * N;

    return style_error + ((weight_sum > 0) ? (guide_error / weight_sum * N) : guide_error);
}

/*
 * NCC cost function logic.
 * This implementation is adapted from the robust Bilateral NCC function found
 * in the ACMH project.
 * It provides better invariance to linear brightness and contrast changes than SSD.
 * https://github.com/GhiXu/ACMH
 */
__device__ float compute_patch_ncc_split(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest,
    bool use_bilateral, float sigma_spatial, float sigma_color, int n_size_step)
{

    const int r = patch_size / 2;
    const float N = patch_size * patch_size;
    const float epsilon = 1e-6f;

    const int num_style_channels = source_style.size(2);
    const int num_guide_channels = source_guide.size(2);
    const int source_h = source_style.size(0);
    const int source_w = source_style.size(1);
    const int target_h = target_style.size(0);
    const int target_w = target_style.size(1);

    // --- NCC for Style ---
    float sum_s = 0.0f, sum_t = 0.0f;
    float sum_sq_s = 0.0f, sum_sq_t = 0.0f;
    float sum_st = 0.0f;
    float style_error = 0.0f;
    float weight_sum = 0.0f;

    for (int py = -r; py <= r; py += n_size_step)
    {
        for (int px = -r; px <= r; px += n_size_step)
        {
            int cur_sx = min(max(sx + px, 0), source_w - 1);
            int cur_sy = min(max(sy + py, 0), source_h - 1);
            int cur_tx = min(max(tx + px, 0), target_w - 1);
            int cur_ty = min(max(ty + py, 0), target_h - 1);

            float s_val = 0.0f, t_val = 0.0f;
            for (int c = 0; c < num_style_channels; ++c)
            {
                s_val += (float)source_style[cur_sy][cur_sx][c];
                t_val += (float)target_style[cur_ty][cur_tx][c];
            }
            s_val /= num_style_channels;
            t_val /= num_style_channels;

            float weight = 1.0f;
            if (use_bilateral)
            {
                float spatial_dist_sq = (float)(px * px + py * py);
                float color_dist_sq = 0.0f;
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float d = (float)source_style[cur_sy][cur_sx][c] - (float)source_style[sy][sx][c];
                    color_dist_sq += d * d;
                }
                weight = expf(-spatial_dist_sq / (2.0f * sigma_spatial * sigma_spatial) - color_dist_sq / (2.0f * sigma_color * sigma_color));
            }
            weight_sum += weight;

            sum_s += weight * s_val;
            sum_t += weight * t_val;
            sum_sq_s += weight * s_val * s_val;
            sum_sq_t += weight * t_val * t_val;
            sum_st += weight * s_val * t_val;
        }
    }

    float mean_s = (weight_sum > 0) ? (sum_s / weight_sum) : 0;
    float mean_t = (weight_sum > 0) ? (sum_t / weight_sum) : 0;
    float var_s = (weight_sum > 0) ? (sum_sq_s / weight_sum - mean_s * mean_s) : 0;
    float var_t = (weight_sum > 0) ? (sum_sq_t / weight_sum - mean_t * mean_t) : 0;
    float std_s = sqrtf(fmaxf(0.0f, var_s));
    float std_t = sqrtf(fmaxf(0.0f, var_t));
    float cov = (weight_sum > 0) ? (sum_st / weight_sum - mean_s * mean_t) : 0;

    float ncc = (std_s > epsilon && std_t > epsilon) ? cov / (std_s * std_t) : 0.0f;
    style_error = (1.0f - ncc) * style_weights[0] * N; // NCC cost, scaled like SSD for compatibility with weights

    // --- SSD for Guides ---
    float guide_error = 0.0f;
    for (int py = -r; py <= r; py += n_size_step)
    {
        for (int px = -r; px <= r; px += n_size_step)
        {
            int cur_sx = min(max(sx + px, 0), source_w - 1);
            int cur_sy = min(max(sy + py, 0), source_h - 1);
            int cur_tx = min(max(tx + px, 0), target_w - 1);
            int cur_ty = min(max(ty + py, 0), target_h - 1);

            float weight = 1.0f;
            if (use_bilateral)
            {
                float spatial_dist_sq = (float)(px * px + py * py);
                float color_dist_sq = 0.0f;
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float d = (float)source_style[cur_sy][cur_sx][c] - (float)source_style[sy][sx][c];
                    color_dist_sq += d * d;
                }
                weight = expf(-spatial_dist_sq / (2.0f * sigma_spatial * sigma_spatial) - color_dist_sq / (2.0f * sigma_color * sigma_color));
            }

            for (int c = 0; c < num_guide_channels; ++c)
            {
                float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
                float modulation = 1.0f;
                if (use_modulation)
                {
                    modulation = (float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f;
                }
                guide_error += weight * guide_weights[c] * modulation * diff * diff;
            }
        }
    }

    return style_error + ((weight_sum > 0) ? (guide_error / weight_sum * N) : guide_error);
}
