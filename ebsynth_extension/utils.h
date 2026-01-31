// ReEzSynth/ebsynth_extension/utils.h
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ===================================================================
//                RNG STATE INITIALIZER KERNEL
// ===================================================================

__global__ void init_rand_states_kernel(curandState* states, int num_states);
