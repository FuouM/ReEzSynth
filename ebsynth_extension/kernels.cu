// ReEzSynth/ebsynth_extension/kernels.cu
// This file is kept for backward compatibility.
// All kernel implementations have been moved to modular .cu files:
//   - cost_functions.cu  (SSD/NCC cost functions)
//   - omega_ops.cu       (Omega map operations)
//   - mask_ops.cu        (Masking kernels)
//   - voting.cu          (Voting kernels)
//   - patchmatch.cu      (PatchMatch core logic)
//   - utils.cu           (Utilities and RNG)
//
// This file now just includes the modular implementations to maintain
// compatibility with existing build systems that compile kernels.cu.

#include "kernels.h"

// Include all modular implementations
// Note: These are included as text, not as separate compilation units
// This maintains the same compilation behavior as the original single-file approach
#include "cost_functions.cu"
#include "omega_ops.cu"
#include "mask_ops.cu"
#include "voting.cu"
#include "patchmatch.cu"
#include "utils.cu"
