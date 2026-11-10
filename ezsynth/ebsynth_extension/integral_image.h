// ReEzSynth/ezsynth/ebsynth_extension/integral_image.h
#pragma once

#include <cuda_runtime.h>

#include "accessor.h"

enum PrepMode
{
    PREP_GRAY,    // Convert to grayscale
    PREP_GRAY_SQR // Convert to grayscale and square
};

// Computes a summed-area table (integral image) of a uint8 HWC image into
// ``output_sat`` (double precision to avoid overflow). ``intermediate`` is a
// device scratch buffer of at least height * width doubles. Work is enqueued on
// ``stream``; returns the launch error, if any.
cudaError_t compute_integral_image_cuda(
    ebs::PackedAccessor64<double, 2> output_sat,
    ebs::PackedAccessor32<uint8_t, 3> input_image,
    double *intermediate,
    PrepMode mode,
    cudaStream_t stream);
