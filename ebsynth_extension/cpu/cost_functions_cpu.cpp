// ReEzSynth/ebsynth_extension/cpu/cost_functions_cpu.cpp
#include "cost_functions_cpu.h"

#include <cmath>
#include <limits>
#include <algorithm>

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD headers
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define REEZ_SIMD_NEON
#elif defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define REEZ_SIMD_AVX2
#endif

// Cost function mode constants
#define COST_FUNCTION_SSD 0
#define COST_FUNCTION_NCC 1

// ===================================================================
//                        SSD COST FUNCTION
// ===================================================================

// ===================================================================
//                        SIMD HELPERS
// ===================================================================

#ifdef REEZ_SIMD_NEON
// Specialized NEON version for 3-channel (RGB) style data
static inline float compute_ssd_row_3ch_neon(const uint8_t* s, const uint8_t* t, int pixels, const float* w) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    float32x4_t w_vec = {w[0], w[1], w[2], 1.0f}; // 4th element unused but safe
    
    int i = 0;
    // Process 1 pixel (3 channels) at a time using NEON
    for (; i < pixels; ++i) {
        uint8x8_t s8 = vld1_u8(s + i * 3);
        uint8x8_t t8 = vld1_u8(t + i * 3);
        
        // Widen to 16-bit, then to 32-bit float
        uint16x8_t s16 = vmovl_u8(s8);
        uint16x8_t t16 = vmovl_u8(t8);
        
        float32x4_t sf = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s16)));
        float32x4_t tf = vcvtq_f32_u32(vmovl_u16(vget_low_u16(t16)));
        
        float32x4_t diff = vsubq_f32(sf, tf);
        float32x4_t sq_diff = vmulq_f32(diff, diff);
        sum_vec = vmlaq_f32(sum_vec, sq_diff, w_vec);
    }
    
    // Horizontal sum of the first 3 elements
    float res[4];
    vst1q_f32(res, sum_vec);
    return res[0] + res[1] + res[2];
}

static inline float compute_ssd_row_gen_neon(const uint8_t* s, const uint8_t* t, int pixels, int channels, const float* w, const uint8_t* mod = nullptr) {
    float total = 0;
    
    // Process pixel-by-pixel to keep channel-to-weight alignment simple
    for (int p = 0; p < pixels; ++p) {
        int base = p * channels;
        float32x4_t p_sum = vdupq_n_f32(0);
        
        int c = 0;
        // Process 4 channels at a time with NEON if available
        for (; c <= channels - 4; c += 4) {
            uint8x8_t s8 = vld1_u8(s + base + c);
            uint8x8_t t8 = vld1_u8(t + base + c);
            uint16x4_t s16 = vget_low_u16(vmovl_u8(s8));
            uint16x4_t t16 = vget_low_u16(vmovl_u8(t8));
            float32x4_t sf = vcvtq_f32_u32(vmovl_u16(s16));
            float32x4_t tf = vcvtq_f32_u32(vmovl_u16(t16));
            float32x4_t diff = vsubq_f32(sf, tf);
            float32x4_t w_f = vld1q_f32(w + c);
            
            float32x4_t term = vmulq_f32(diff, diff);
            if (mod) {
                float32x4_t m_f = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(mod + base + c))))), 1.0f/255.0f);
                term = vmulq_f32(term, m_f);
            }
            p_sum = vmlaq_f32(p_sum, term, w_f);
        }
        
        total += vaddvq_f32(p_sum);
        
        // Remainder
        for (; c < channels; ++c) {
            float d = (float)s[base + c] - (float)t[base + c];
            float m = mod ? ((float)mod[base + c] / 255.0f) : 1.0f;
            total += w[c] * m * d * d;
        }
    }
    return total;
}
#endif

#ifdef REEZ_SIMD_AVX2
// AVX2 version for 3-channel (RGB) style data
static inline float compute_ssd_row_3ch_avx2(const uint8_t* s, const uint8_t* t, int pixels, const float* w) {
    __m128 sum_vec = _mm_setzero_ps();
    __m128 w_vec = _mm_setr_ps(w[0], w[1], w[2], 0.0f);

    for (int i = 0; i < pixels; ++i) {
        // Load 3 bytes safely
        uint32_t s32 = 0, t32 = 0;
        memcpy(&s32, s + i * 3, 3);
        memcpy(&t32, t + i * 3, 3);
        
        __m128i s_vec_i = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(s32));
        __m128i t_vec_i = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(t32));
        
        __m128 sf = _mm_cvtepi32_ps(s_vec_i);
        __m128 tf = _mm_cvtepi32_ps(t_vec_i);
        
        __m128 diff = _mm_sub_ps(sf, tf);
        __m128 sq_diff = _mm_mul_ps(diff, diff);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(sq_diff, w_vec));
    }
    
    // Horizontal sum
    float res[4];
    _mm_storeu_ps(res, sum_vec);
    return res[0] + res[1] + res[2];
}

static inline float compute_ssd_row_gen_avx2(const uint8_t* s, const uint8_t* t, int pixels, int channels, const float* w, const uint8_t* mod = nullptr) {
    float total = 0;

    for (int p = 0; p < pixels; ++p) {
        int base = p * channels;
        __m256 p_sum_vec = _mm256_setzero_ps();
        
        int c = 0;
        // Process 8 channels at a time with AVX2 if available
        for (; c <= channels - 8; c += 8) {
            // Load 8 uint8 bytes
            __m128i s8 = _mm_loadl_epi64((const __m128i*)(s + base + c));
            __m128i t8 = _mm_loadl_epi64((const __m128i*)(t + base + c));

            // Convert to 32-bit integers (AVX2)
            __m256i s32 = _mm256_cvtepu8_epi32(s8);
            __m256i t32 = _mm256_cvtepu8_epi32(t8);

            // Convert to 32-bit floats
            __m256 sf = _mm256_cvtepi32_ps(s32);
            __m256 tf = _mm256_cvtepi32_ps(t32);

            __m256 diff = _mm256_sub_ps(sf, tf);
            __m256 sq_diff = _mm256_mul_ps(diff, diff);
            
            // Weights
            __m256 w_f = _mm256_loadu_ps(w + c);
            
            __m256 term = _mm256_mul_ps(sq_diff, w_f);
            
            if (mod) {
                __m128i m8 = _mm_loadl_epi64((const __m128i*)(mod + base + c));
                __m256i m32 = _mm256_cvtepu8_epi32(m8);
                __m256 mf = _mm256_cvtepi32_ps(m32);
                // Divide by 255.0f
                mf = _mm256_mul_ps(mf, _mm256_set1_ps(1.0f/255.0f));
                term = _mm256_mul_ps(term, mf);
            }
            
            p_sum_vec = _mm256_add_ps(p_sum_vec, term);
        }
        
        // Sum up the vector
        float res[8];
        _mm256_storeu_ps(res, p_sum_vec);
        for(int k=0; k<8; ++k) total += res[k];
        
        // Remainder
        for (; c < channels; ++c) {
            float d = (float)s[base + c] - (float)t[base + c];
            float m = mod ? ((float)mod[base + c] / 255.0f) : 1.0f;
            total += w[c] * m * d * d;
        }
    }
    return total;
}
#endif

float compute_patch_ssd_split_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest)
{

    const int r = patch_size / 2;
    float error = 0.0f;

    const int num_style_channels = source_style.size(2);
    const int num_guide_channels = source_guide.size(2);

    const int source_h = source_style.size(0);
    const int source_w = source_style.size(1);
    const int target_h = target_style.size(0);
    const int target_w = target_style.size(1);

    // Get raw pointers for faster access
    const uint8_t* s_style_ptr = source_style.data();
    const uint8_t* t_style_ptr = target_style.data();
    const uint8_t* s_guide_ptr = source_guide.data();
    const uint8_t* t_guide_ptr = target_guide.data();
    const uint8_t* t_mod_ptr = use_modulation ? target_modulation_guide.data() : nullptr;
    
    // Strides
    const int s_style_s0 = source_style.stride(0);
    const int s_style_s1 = source_style.stride(1);
    const int t_style_s0 = target_style.stride(0);
    const int t_style_s1 = target_style.stride(1);
    
    const int s_guide_s0 = source_guide.stride(0);
    const int s_guide_s1 = source_guide.stride(1);
    const int t_guide_s0 = target_guide.stride(0);
    const int t_guide_s1 = target_guide.stride(1);
    const int t_mod_s0 = use_modulation ? target_modulation_guide.stride(0) : 0;
    const int t_mod_s1 = use_modulation ? target_modulation_guide.stride(1) : 0;

    const float* s_w = style_weights.data();
    const float* g_w = guide_weights.data();

    // Use pointer arithmetic for faster access when within bounds
    const bool in_bounds = (sx - r >= 0 && sx + r < source_w && sy - r >= 0 && sy + r < source_h &&
                            tx - r >= 0 && tx + r < target_w && ty - r >= 0 && ty + r < target_h);

    if (in_bounds)
    {
        // Fast path: all pixels are within bounds
        for (int py = -r; py <= r; ++py)
        {
            const int cur_sy = sy + py;
            const int cur_ty = ty + py;
            const int cur_sx_start = sx - r;
            const int cur_tx_start = tx - r;

            const uint8_t* s_row = s_style_ptr + cur_sy * s_style_s0 + cur_sx_start * s_style_s1;
            const uint8_t* t_row = t_style_ptr + cur_ty * t_style_s0 + cur_tx_start * t_style_s1;

#if defined(REEZ_SIMD_NEON)
            if (num_style_channels == 3) {
                error += compute_ssd_row_3ch_neon(s_row, t_row, patch_size, s_w);
            } else {
                error += compute_ssd_row_gen_neon(s_row, t_row, patch_size, num_style_channels, s_w);
            }
#elif defined(REEZ_SIMD_AVX2)
            if (num_style_channels == 3) {
                error += compute_ssd_row_3ch_avx2(s_row, t_row, patch_size, s_w);
            } else {
                error += compute_ssd_row_gen_avx2(s_row, t_row, patch_size, num_style_channels, s_w);
            }
#else
            for (int px = 0; px < patch_size; ++px)
            {
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float diff = (float)s_row[px*num_style_channels+c] - (float)t_row[px*num_style_channels+c];
                    error += s_w[c] * diff * diff;
                }
            }
#endif

            // Guide difference
            const uint8_t* sg_row = s_guide_ptr + cur_sy * s_guide_s0 + cur_sx_start * s_guide_s1;
            const uint8_t* tg_row = t_guide_ptr + cur_ty * t_guide_s0 + cur_tx_start * t_guide_s1;
            const uint8_t* tm_row = use_modulation ? (t_mod_ptr + cur_ty * t_mod_s0 + cur_tx_start * t_mod_s1) : nullptr;

#if defined(REEZ_SIMD_NEON)
            error += compute_ssd_row_gen_neon(sg_row, tg_row, patch_size, num_guide_channels, g_w, tm_row);
#elif defined(REEZ_SIMD_AVX2)
            error += compute_ssd_row_gen_avx2(sg_row, tg_row, patch_size, num_guide_channels, g_w, tm_row);
#else
            for (int px = 0; px < patch_size; ++px)
            {
                for (int c = 0; c < num_guide_channels; ++c)
                {
                    float diff = (float)sg_row[px*num_guide_channels+c] - (float)tg_row[px*num_guide_channels+c];
                    float modulation = 1.0f;
                    if (use_modulation)
                    {
                        modulation = (float)tm_row[px*num_guide_channels+c] / 255.0f;
                    }
                    error += g_w[c] * modulation * diff * diff;
                }
            }
#endif

            if (error > ebest)
                return error;
        }
    }
    else
    {
        // Slow path: need bounds checking
        for (int py = -r; py <= r; ++py)
        {
            for (int px = -r; px <= r; ++px)
            {
                int cur_sx = std::min(std::max(sx + px, 0), source_w - 1);
                int cur_sy = std::min(std::max(sy + py, 0), source_h - 1);
                int cur_tx = std::min(std::max(tx + px, 0), target_w - 1);
                int cur_ty = std::min(std::max(ty + py, 0), target_h - 1);

                // Style difference
                for (int c = 0; c < num_style_channels; ++c)
                {
                    float diff = (float)source_style[cur_sy][cur_sx][c] - (float)target_style[cur_ty][cur_tx][c];
                    error += style_weights[c] * diff * diff;
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
                    error += guide_weights[c] * modulation * diff * diff;
                }
            }
            if (error > ebest)
                return error;
        }
    }
    return error;
}

// ===================================================================
//                        NCC COST FUNCTIONS
// ===================================================================

static double query_sat_cpu(
    torch::PackedTensorAccessor64<double, 2> sat,
    int x1, int y1, int x2, int y2)
{

    const int h = sat.size(0);
    const int w = sat.size(1);

    x1 = std::max(x1, 0);
    y1 = std::max(y1, 0);
    x2 = std::min(x2, w - 1);
    y2 = std::min(y2, h - 1);

    double br = sat[y2][x2];
    double bl = (x1 > 0) ? sat[y2][x1 - 1] : 0.0;
    double tr = (y1 > 0) ? sat[y1 - 1][x2] : 0.0;
    double tl = (x1 > 0 && y1 > 0) ? sat[y1 - 1][x1 - 1] : 0.0;

    return br - bl - tr + tl;
}

float compute_patch_ncc_sat_cpu(
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
    torch::PackedTensorAccessor64<double, 2> target_style_sq_sat)
{

    const int r = patch_size / 2;
    const float N = patch_size * patch_size;
    const float epsilon = 1e-6f;

    const int num_style_channels = source_style.size(2);
    const int num_guide_channels = source_guide.size(2);

    // --- O(1) Style Stats using SATs ---
    double sum_s = query_sat_cpu(source_style_sat, sx - r, sy - r, sx + r, sy + r);
    double sum_sq_s = query_sat_cpu(source_style_sq_sat, sx - r, sy - r, sx + r, sy + r);
    double sum_t = query_sat_cpu(target_style_sat, tx - r, ty - r, tx + r, ty + r);
    double sum_sq_t = query_sat_cpu(target_style_sq_sat, tx - r, ty - r, tx + r, ty + r);

    double mean_s = sum_s / N;
    double mean_t = sum_t / N;
    double std_s = std::sqrt(std::max(0.0, sum_sq_s / N - mean_s * mean_s));
    double std_t = std::sqrt(std::max(0.0, sum_sq_t / N - mean_t * mean_t));

    const int source_h = source_style.size(0);
    const int source_w = source_style.size(1);
    const int target_h = target_style.size(0);
    const int target_w = target_style.size(1);

    // --- O(P^2) Cross-correlation and Guide SSD ---
    double sum_st = 0.0;
    float guide_error = 0.0f;

    for (int py = -r; py <= r; ++py)
    {
        for (int px = -r; px <= r; ++px)
        {
            int cur_sx = std::min(std::max(sx + px, 0), source_w - 1);
            int cur_sy = std::min(std::max(sy + py, 0), source_h - 1);
            int cur_tx = std::min(std::max(tx + px, 0), target_w - 1);
            int cur_ty = std::min(std::max(ty + py, 0), target_h - 1);

            // Cross-correlation term
            float s_val_g = 0.0f, t_val_g = 0.0f;
            for (int c = 0; c < num_style_channels; ++c)
            {
                s_val_g += (float)source_style[cur_sy][cur_sx][c];
                t_val_g += (float)target_style[cur_ty][cur_tx][c];
            }
            sum_st += (s_val_g / num_style_channels) * (t_val_g / num_style_channels);

            // Guide difference (SSD)
            for (int c = 0; c < num_guide_channels; ++c)
            {
                float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
                float modulation = use_modulation ? ((float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f) : 1.0f;
                guide_error += guide_weights[c] * modulation * diff * diff;
            }
        }
    }

    double cov = sum_st / N - mean_s * mean_t;
    float ncc = (std_s > epsilon && std_t > epsilon) ? cov / (std_s * std_t) : 0.0f;
    float style_error = (1.0f - ncc) * style_weights[0] * N;

    return style_error + guide_error;
}

float compute_patch_ncc_split_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> source_style,
    torch::PackedTensorAccessor32<uint8_t, 3> target_style,
    torch::PackedTensorAccessor32<uint8_t, 3> source_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_guide,
    torch::PackedTensorAccessor32<uint8_t, 3> target_modulation_guide,
    bool use_modulation,
    int sx, int sy, int tx, int ty, int patch_size,
    const torch::PackedTensorAccessor32<float, 1> style_weights,
    const torch::PackedTensorAccessor32<float, 1> guide_weights,
    float ebest)
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

    for (int py = -r; py <= r; ++py)
    {
        for (int px = -r; px <= r; ++px)
        {
            int cur_sx = std::min(std::max(sx + px, 0), source_w - 1);
            int cur_sy = std::min(std::max(sy + py, 0), source_h - 1);
            int cur_tx = std::min(std::max(tx + px, 0), target_w - 1);
            int cur_ty = std::min(std::max(ty + py, 0), target_h - 1);

            float s_val = 0.0f, t_val = 0.0f;
            for (int c = 0; c < num_style_channels; ++c)
            {
                s_val += (float)source_style[cur_sy][cur_sx][c];
                t_val += (float)target_style[cur_ty][cur_tx][c];
            }
            s_val /= num_style_channels;
            t_val /= num_style_channels;

            sum_s += s_val;
            sum_t += t_val;
            sum_sq_s += s_val * s_val;
            sum_sq_t += t_val * t_val;
            sum_st += s_val * t_val;
        }
    }

    float mean_s = sum_s / N;
    float mean_t = sum_t / N;
    float std_s = std::sqrt(std::max(0.0f, sum_sq_s / N - mean_s * mean_s));
    float std_t = std::sqrt(std::max(0.0f, sum_sq_t / N - mean_t * mean_t));
    float cov = sum_st / N - mean_s * mean_t;

    float ncc = (std_s > epsilon && std_t > epsilon) ? cov / (std_s * std_t) : 0.0f;
    float style_error = (1.0f - ncc) * style_weights[0] * N;

    // --- SSD for Guides ---
    float guide_error = 0.0f;
    for (int py = -r; py <= r; ++py)
    {
        for (int px = -r; px <= r; ++px)
        {
            int cur_sx = std::min(std::max(sx + px, 0), source_w - 1);
            int cur_sy = std::min(std::max(sy + py, 0), source_h - 1);
            int cur_tx = std::min(std::max(tx + px, 0), target_w - 1);
            int cur_ty = std::min(std::max(ty + py, 0), target_h - 1);

            for (int c = 0; c < num_guide_channels; ++c)
            {
                float diff = (float)source_guide[cur_sy][cur_sx][c] - (float)target_guide[cur_ty][cur_tx][c];
                float modulation = 1.0f;
                if (use_modulation)
                {
                    modulation = (float)target_modulation_guide[cur_ty][cur_tx][c] / 255.0f;
                }
                guide_error += guide_weights[c] * modulation * diff * diff;
            }
        }
    }

    return style_error + guide_error;
}
