// ReEzSynth/ebsynth_extension/utils.cu
#include "utils.h"

// ===================================================================
//                RNG STATE INITIALIZER KERNEL
// ===================================================================

__global__ void init_rand_states_kernel(curandState* states, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(1337, idx, 0, &states[idx]);
    }
}
