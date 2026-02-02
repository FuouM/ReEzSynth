// ReEzSynth/ebsynth_extension/cpu/integral_image_cpu.h
#pragma once

#include <torch/extension.h>

// Prep mode constants (must match integral_image.h)
enum PrepMode
{
    PREP_GRAY,    // Convert to grayscale
    PREP_GRAY_SQR // Convert to grayscale and square
};

// Computes a summed-area table (integral image) from a uint8 image.
// The SAT itself will be a double precision tensor to avoid overflow.
void compute_integral_image_cpu(
    torch::Tensor output_sat,
    torch::Tensor input_image,
    PrepMode mode);
