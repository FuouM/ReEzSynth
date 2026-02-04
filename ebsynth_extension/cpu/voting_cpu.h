// ReEzSynth/ebsynth_extension/cpu/voting_cpu.h
#pragma once

#include <torch/extension.h>
#include <cstdint>

// Vote mode constants (must match voting.h)
#define EBSYNTH_VOTEMODE_PLAIN 0x0001
#define EBSYNTH_VOTEMODE_WEIGHTED 0x0002

// Helper for vector types
template <int N, typename T>
struct Vec
{
    T v[N];
};

// ===================================================================
//                        VOTING OPERATIONS
// ===================================================================

struct VoteAccumulator
{
    float sumColor[4];
    float sumWeight;
};

// Plain voting - uniform weights
void krnlVotePlain_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    int patch_size,
    int target_h, int target_w);

// Weighted voting - weights based on patch error
void krnlVoteWeighted_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    int patch_size,
    int target_h, int target_w);

// Incremental voting - delta updates
void krnlVoteIncremental_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> target,
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf_current,
    torch::PackedTensorAccessor32<int32_t, 3> nnf_prev,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<float, 3> accumulators,
    int patch_size,
    int target_h, int target_w);

// Populate persistent accumulators from scratch
void krnlVotePopulate_cpu(
    torch::PackedTensorAccessor32<uint8_t, 3> source,
    torch::PackedTensorAccessor32<int32_t, 3> nnf,
    torch::PackedTensorAccessor32<float, 2> error_map,
    torch::PackedTensorAccessor32<float, 3> accumulators,
    int patch_size,
    int target_h, int target_w);
