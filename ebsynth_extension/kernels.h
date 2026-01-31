// ReEzSynth/ebsynth_extension/kernels.h
// This file is kept for backward compatibility.
// All kernel declarations have been moved to modular headers:
//   - cost_functions.h  (SSD/NCC cost functions)
//   - omega_ops.h       (Omega map operations)
//   - mask_ops.h        (Masking kernels)
//   - voting.h          (Voting kernels)
//   - patchmatch.h      (PatchMatch core logic)
//   - utils.h           (Utilities and RNG)
//
// New code should include the specific headers directly.

#pragma once

// Re-export all modular headers for convenience
#include "cost_functions.h"
#include "omega_ops.h"
#include "mask_ops.h"
#include "voting.h"
#include "patchmatch.h"
#include "utils.h"

// Also include integral_image for completeness
#include "integral_image.h"
