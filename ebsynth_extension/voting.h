// ReEzSynth/ebsynth_extension/voting.h
#pragma once

#include <cuda_runtime.h>

#include "accessor.h"

// Vote mode constants
#define EBSYNTH_VOTEMODE_PLAIN 0x0001
#define EBSYNTH_VOTEMODE_WEIGHTED 0x0002

// Helper for CUDA vector types
template <int N, typename T>
struct Vec
{
  T v[N];
};

// ===================================================================
//                        VOTING KERNELS
// ===================================================================

// Plain voting - uniform weights
__global__ void krnlVotePlain(
    ebs::PackedAccessor32<uint8_t, 3> target,
    ebs::PackedAccessor32<uint8_t, 3> source,
    ebs::PackedAccessor32<int32_t, 3> nnf,
    int patch_size);

// Weighted voting - weights based on patch error
__global__ void krnlVoteWeighted(
    ebs::PackedAccessor32<uint8_t, 3> target,
    ebs::PackedAccessor32<uint8_t, 3> source,
    ebs::PackedAccessor32<int32_t, 3> nnf,
    ebs::PackedAccessor32<float, 2> error_map,
    int patch_size);
