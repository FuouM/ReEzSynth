# Deep Implementation Guide: Porting Spatial Algorithms to Taichi

> *Reference for AI Agents and Systems Engineers*
>
> <https://github.com/taichi-dev/taichi> - Github Repository
>
> <https://docs.taichi-lang.org/> - Docs
>
> **Quick Reference IDs**: `GUIDE-ARCH`, `GUIDE-KERNEL`, `GUIDE-NUMERIC`, `GUIDE-HARDWARE`, `GUIDE-DEBUG`, `GUIDE-PERF`, `GUIDE-FACEBLIT`, `GUIDE-EBSYNTH`

This guide provides an exhaustive deep-dive into the architectural patterns and numerical pitfalls encountered while porting **EBSynth** (a complex PatchMatch-based spatial consistency algorithm) from C++/CUDA to Taichi.

---

## 1. Architectural Strategy: The "Agentic Porting Pattern"

`GUIDE-ARCH`

When porting a legacy C++ codebase, agents should avoid direct line-by-line translation and instead follow the **Data-Oriented Class Pattern**.

### 1.1 Why `@ti.data_oriented`?

By structuring the backend as a Python class decorated with `@ti.data_oriented`, you gain:

- **Managed "this" Context**: Taichi kernels can access class instance variables during compilation.
- **Kernel Modularization**: You can split massive logic into logical `ti.kernel` and `ti.func` blocks without global naming conflicts.
- **Lazy Compilation Optimization**: Since you initialize the backend once, Taichi compiles kernels on the first call, allowing subsequent runs to be near-instant.

### 1.2 Prefer `ti.types.ndarray` over `ti.field`

For agents porting PyTorch-based libraries: **Internal Fields are usually a mistake.**

- **Fields**: Require manual memory management (`field.from_numpy`, etc.) and complicate the data pipeline.
- **Ndarrays**: Directly mapped to PyTorch tensors in-place. Use `ti.types.ndarray()` in kernel signatures. This allows the agent to handle high-level logic in Python/PyTorch (resizing, masking) and low-level intensity in Taichi.

---

## 2. Advanced Kernel Implementation Patterns

`GUIDE-KERNEL`

### 2.1 The "Thread-Local Context" Pattern

Taichi's `for i, j in ti.ndrange(...)` is a massively parallel construct (Grid-level in CUDA terms). However, algorithms like PatchMatch require sequential dependencies within a logical "pixel task."

**Anti-pattern**: Trying to split propagation into multiple kernels.
**Design Pattern**: Wrap sequential logic inside a single thread's execution path.

```python
@ti.kernel
def sequential_task_kernel(data: ti.types.ndarray()):
    for y_it, x_it in ti.ndrange(H, W):
        # Sequential logic for a single pixel (e.g., propagation)
        best_match = ...
        for step in [-1, 1]: # sequential loop inside a parallel thread
             cand = try_neighbor(..., step)
             if better(cand, best_match):
                 best_match = cand
```

### 2.2 The `ti.template()` Dependency

When building modular kernels, you will use `@ti.func` for helper utilities.

**Critical Rule**: If a `@ti.func` needs to access a tensor passed to the parent kernel, that argument must be tagged `ti.template()`.

- Without `ti.template()`, Taichi will try to evaluate the array as a compile-time constant or throw a DIM mismatch.
- **SAT Example**: `def query_sat(sat: ti.template(), ...)` ensures the SAT buffer is passed by reference to the device function.
- **Multi-Return Constraint**: Taichi kernels/funcs do NOT support multiple `return` statements inside non-static `if/else` blocks.
- **Pattern**: Use a `final_val = 0.0` accumulator and a single return at the end of the function to avoid `Return inside non-static if/for is not supported` errors.

### 2.3 The "Manual Step" Iteration Pattern

Taichi's `range()` and `ndrange()` do not consistently support a 3rd `step` argument across all backends (especially Metal).

- **Problem**: `for i in range(start, end, step)` throws a compilation error on many targets.
- **Fix**: Calculate the number of steps in Python/PyTorch, pass that to the kernel, and use a normalized loop.

```python
steps = (2 * radius) // step_size + 1
for i, j in ti.ndrange(steps, steps):
    px = -radius + i * step_size
    py = -radius + j * step_size
    # px and py now have the correct sparse coordinates
```

---

## 3. Numerical Stability "Death Valley"

`GUIDE-NUMERIC`

Getting "working" results is easy; getting "identical" results is where agents fail.

### 3.1 Integer Truncation vs. Floor

- **C++/CUDA `int(a * b)`**: Truncates towards zero.
- **Python/Taichi `int()`**: Be mindful of negative numbers.
- **Reference Logic**: In Random Search, where offsets are `[-r, r]`, use `ti.floor(ti.random() * (2*r + 1)) - r`. This ensures an exactly balanced distribution including the endpoints, matching `curand % (2r+1) - r`.

### 3.2 Distance Metric Parity (SSD/NCC)

Agents often hallucinate "standard" formulas. ReEzSynth/EBSynth has specific twists:

- **Max Absolute Difference**: Used for convergence masking. `max(abs(R1-R2), abs(G1-G2), ...)` is NOT the same as RMS or SSD. Check the source `mask_ops.cu` every time.
- **Floating Point Accumulation**: Summing squared `uint8` values into `float32`.
  - **CUDA**: Uses `__fadd_rn` (round to nearest).
  - **Taichi/Metal**: Ensure you don't add a `0.5` rounding bias during reconstruction (`voting`) unless the reference uses `round()`. Reference EBSynth uses binary truncation.

### 3.3 The Coordinate "Identity Crisis" (Padding & Clamping)

One of the most frequent sources of "slightly different" output is how boundaries are handled.

- **C++ (CUDA)**: Usually iterates over a 0-indexed grid and manually offsets indices.
- **PyTorch**: Uses `F.pad` (constant, replicate, reflect).
- **Taichi Discovery**: When porting, I found that using `ti.max(0, ti.min(x, W-1))` inside the kernel is numerically safer than relying on pre-padded PyTorch tensors, as it ensures memory indices never escape the physical buffer during large-radius random searches.
- **Off-by-One**: In PatchMatch, a patch of size `P` has radius `R = P // 2`. Ensure your loop bounds are `[-R, R+1]` (inclusive of the lower bound, exclusive of the upper in Taichi's range).

### 3.4 The "Invisible" Scaling Pitfall

In spatial algorithms like PatchMatch, "Normalization" is often implemented implicitly in CUDA but must be explicit in Taichi.

- **The Case of Omega**: In the reference code, a penalty might be described as "per pixel," but the NNF treats it as "per patch."
- **Agent Discovery**: If your output is "more noisy" than the reference, check if you're missing a division by $P^2$ (patch area). This simple scaling error made my initial Taichi results visually distinct from CUDA, even though the logic felt "correct."

---

## 4. Hardware Specific Constraints

`GUIDE-HARDWARE`

### 4.1 Hardware Precision (The M4/Metal Wall)

- **Problem**: Metal (Apple Silicon) does not support `f64` (float64/double).
- **Fallout**: If porting a CUDA algorithm that relies on `double` for high-precision Integral Images, the agent **must** switch to `float32` and accept a minor numerical drift, or implement a custom "Block-based SAT" to mitigate overflow.

### 4.2 Metal Atomic Operations (The 32-bit Limit)

Metal hardware on Apple Silicon **does not support 64-bit atomics** (e.g. `ti.atomic_min` on an `i64`).

- **Agent Discovery**: If your algorithm needs to atomically update a `(cost, coordinate)` pair:
  - **CUDA**: You can pack into a 64-bit integer `(u64)`.
  - **Metal**: You MUST pack into a **32-bit integer** `(i32)`.
- **Strategy**: Use "Lossy Bit-Packing" (see Section 6).

### 4.3 Deciphering Metal/SPIR-V Stack Traces

Agents don't get pretty Python errors when Metal fails.

- **"Type f64 not supported"**: This is often buried 50 lines deep in a C++ traceback. It means you passed a `torch.float64` tensor.
- **"Array with dim 1 accessed with indices of dim 3"**: This happens if you use `modulation_guide[y, x, c]` but passed a 1D dummy tensor for an optional argument.
- **Agent Strategy**: Optional tensors (like modulation masks) should be passed as **dummy 3D tensors** (e.g., `(1, 1, 1)`) rather than empty tensors to satisfy the Taichi compiler's type-inference, even if the branch is protected by an `if use_modulation:`.

---

## 5. Visual & Numerical Debugging Strategy

`GUIDE-DEBUG`

Since agents cannot "see" the image, use this tiered verification:

1. **Phase 1: Bit-Identical SSD**: Force PyTorch and Taichi to use the EXACT same coordinate-clamping and truncation. You should reach **0.000 max difference**. If it's `1e-7`, it's floating point. If it's `1.0`, it's a boundary pixel or rounding bias.
2. **Phase 2: Distribution Parity**: If SSD matches but the full loop fails, the error is in **Random Search**. Calculate the mean/std of your random offsets over 1M iterations. If not centered at 0.0, your `ti.random()` logic is biased.
3. **Phase 3: The Voting Truncation**: Many image processing kernels use `static_cast<uint8_t>(val)` in C++. In Taichi, `ti.u8(sum_color / sum_weight)` behaves identically. **Do not add 0.5** for "better rounding" unless you specifically see `round()` in the reference.

---

## 6. Performance Optimization for Agents

`GUIDE-PERF`

### 6.1 The Summed-Area Table (SAT) Hack

For algorithms that compute stats over patches (like NCC), patch-wise iteration is `O(H * W * P^2)`.
**Porting strategy**: Always implement SATs.

1. Kernel 1: Horizontal prefix sum.
2. Kernel 2: Vertical prefix sum.
3. Result: Statistical queries in `O(1)`.
In Taichi, these two kernels are faster than a single PyTorch `cumsum` because they keep data on-device and avoid the Python overhead.

### 6.2 High-Resolution Spatial Packing (The 31-bit Pattern)

When packing `(cost, x, y)` into an `i32` for `atomic_min` on Metal, every bit counts.

1. **Coordinate space**: 1024x1024 requires **10 bits per dimension**.
2. **Signedness**: Stay within **31 bits** (positive range) to ensure `atomic_min` treats the integer as an unsigned comparison.
3. **Error Scaling**: 11 bits for cost (0-2047). Scale your error by 4 or 8 to fit.

```python
# The "FaceBlit Pattern" for 1024x1024 support
packed_val = (ti.i32(ti.min(err // 4, 2047)) << 20) | (ti.i32(sx) << 10) | ti.i32(sy)
ti.atomic_min(lut[tx, ty, z], packed_val)
```

### 6.3 Parallel Mode Filter (NNF Denoising)

Taichi doesn't have a built-in `mode()` (histogram) for patches.

- **Agent Discovery**: For small patches (3x3), a nested brute-force loop inside each thread is surprisingly efficient on GPU.
- **Pattern**: Instead of building a complex histogram, for each pixel, iterate the patch twice: once to pick a candidate offset, once to count its frequency. This avoids dynamic memory allocation which is forbidden in kernels.

### 6.4 The "Inner Break" Flag Pattern

While Taichi supports `break`, using deep `break` statements from nested loops on GPU backends can occasionally lead to non-deterministic behavior or compilation failure if the logic is too complex.

- **Discovery**: When implementing SSD with early exit, use a `break_flag` variable to propagate the exit signal across nested spatial loops if you encounter performance degradation or compiler hangs.

---

## 7. Engineering Chronicles: The FaceBlit Port (Bug Log)

`GUIDE-FACEBLIT`

Porting FaceBlit from a PyTorch vectorized implementation to Taichi exposed several deep-tissue hardware and software issues.

### 7.1 The "Infinity" Initialization Crash

**Symptom**: `RuntimeError: value cannot be converted to type int64_t without overflow`

- **Cause**: Trying to initialize a `torch.int64` tensor with `(1 << 63)` in Python for use as an "infinity" cost. Python handles large ints, but the torch C++ backend rejects values that clip the signed 64-bit boundary.
- **Fix**: Use `(1 << 62)` or a safe 32-bit infinity like `(1 << 30)` if using 32-bit packing.
- **Lesson**: Don't use hardware-max integers for sentinel values; leave a 1-bit safety margin.

### 7.2 The "Backend Leak" Syntax Error

**Symptom**: `Taichi data types cannot be called outside Taichi kernels.`

- **Bug**: An agent might try to use `ti.i64(k_inf)` in standard Python code (e.g., in a class constructor or setup method).
- **Fix**: Use standard Python types (`int`, `float`) for arguments passed *to* kernels. Taichi handles the casting at the kernel boundary.

### 7.3 The Metal Atomic Wall (The "Showstopper")

**Symptom**: `RHI Error: MSL currently does not support 64-bit atomics.`

- **Observation**: Taichi code using `ti.atomic_min` on 64-bit packed integers (`i64`) works on CUDA but **aborts** on Metal (Apple Silicon).
- **Hard Limit**: Metal hardware simply does not provide 64-bit atomic guarantee.
- **Fix**: This forced the transition from `(err << 32 | sx << 16 | sy)` to a **31-bit packing scheme** to stay within the positive range of a signed `int32`.

### 7.4 Coordinate Aliasing (Tiling Artifacts)

**Symptom**: Stylized output looked like a 256px tiled grid of the style face.

- **Math Error**: Initial packing used 8 bits for X and Y coordinates. $2^8 = 256$.
- **Problem**: Real-world images (like the 1024x768 example) have coordinates $> 255$. When truncated to 8 bits, `coordinate % 256` created a periodic tiling effect.
- **Fix**: Re-allocate bit-depth. 10 bits for coordinates ($2^{10} = 1024$) and 11 bits for cost.

### 7.5 The "Red/Green" Channel Swap

**Symptom**: Output face was recognizable but "scrambled" or shifted.

- **Bug**: The FaceBlit Positional Guide uses Red for X and Green for Y. If the kernel maps `tr -> y` and `tg -> x`, the lookup logic will be rotated/mirrored.
- **Fix**: Explicitly document channel mapping: `tr (Red) = X-axis`, `tg (Green) = Y-axis`. Ensure consistency between the Search kernel (creating the LUT) and the Lookup kernel (reading the LUT).

### 7.6 The Metal Segfault & `TI_DEBUG`

**Symptom**: `zsh: segmentation fault` when running complex kernels (Mode Filter).

- **Mystery**: The code worked with `TI_DEBUG=1` but crashed in production.
- **Insight**: Metal's out-of-bounds checking is not supported, but `TI_DEBUG` changes how the compiler optimizes the SPIR-V.
- **Stability Fix**: Avoid deep nested loops where possible and ensure all `ti.ndrange` bounds are strictly checked. If a segfault persists, check for **stack overflow** inside kernels with very deep local recursions or massive unrolled loops.

---

## 8. Engineering Chronicles: The EBSynth Port

`GUIDE-EBSYNTH`

This port focused on high-precision spatial consistency and revealed many "logic translations" that agents often miss.

### 8.1 The "Implicit Initialization" Trap

**Symptom**: SSD results matched for 1 iteration, then drifted wildly.

- **Cause**: The PyTorch backend used `F.interpolate` (bilinear) for the initial target style, while the native CUDA code often uses a "Initial Vote" pass from the initial NNF.
- **Fix**: Replicated the `initial_vote` pass in the `run_level` loop before entering the search-vote main loop.
- **Lesson**: Never assume the "Initial State" of an algorithm is a simple zero-tensor or a standard resize. Check the initialization pipeline.

### 8.2 The `ti.template()` ndarray Trap

**Symptom**: `TypeError: Invalid constant scalar data type` when passing a tensor to a helper function.

- **Cause**: Taichi `@ti.func` treat arguments as scalars by default. If you pass a `ti.types.ndarray()`, you MUST use `ti.template()` as the type hint.
- **Fix**: `def helper(arr: ti.template(), ...)`
- **Lesson**: Devices functions need templates to "see" the memory layout of external arrays.

### 8.3 The SAT Precision Drift (Metal)

**Symptom**: NCC results showed a max difference of ~270.0 in verification.

- **Hard Limit**: Metal (M4) does not support `float64`. Calculating Summed-Area Tables (SAT) for squared pixel values requires massive dynamic range.
- **Result**: In `f32`, summing $255^2$ over a 1024x1024 image exceeds the mantissa precision of a 32-bit float ($2^{23}$ bits).
- **Strategy**: For agents, if you see drift in SAT-based NCC on Metal, it's a hardware trait. Accept minor drift or implement block-based summation.

### 8.4 The `1/P^2` Scaling Ghost

**Symptom**: "Output is slightly different" (User report).

- **Bug**: The uniformity penalty $U \cdot \frac{\omega}{\omega_{best}}$ should be averaged per patch pixel. If you miss the `1 / patch_size**2` factor, the penalty is 49x too strong (for 7x7 patches).
- **Fix**: Updated `omega_best` to include the `(patch_size * patch_size)` multiplier.
- **Lesson**: "Normalization" in CUDA papers is often shorthand for "divide by patch area."

### 8.5 The Stop-Threshold Unit Mismatch

**Symptom**: Synthesis stopped too early or didn't stop at all.

- **Bug**: Python config passed `stop_threshold=0.01`, but the CUDA kernel expected an integer `[0-255]`.
- **Fix**: Multiplied the threshold by 255 before passing to the kernel.
- **Lesson**: Trace every config variable from the UI/Config object all the way to the hardware register.

### 8.6 Propagation Direction Flip

**Symptom**: Artifacts appeared on the right/bottom edges.

- **Bug**: In PatchMatch, a Backward pass (Iter 0, 2...) should iterate $H-1 \to 0$ and look at Right/Bottom neighbors. My initial Taichi code had the loop direction correct, but the neighbor lookup was looking Left/Top.
- **Fix**: Synchronized the `step` direction with the `is_odd` iteration flag.
- **Lesson**: The NNF is a "living" field; the order of operations creates the path for information flow.

---

## 9. Summary for the Next Agent

You are porting code that has been optimized for a specific hardware (NVIDIA GPU). Your job is to extract the **intent** of that optimization and translate it into Taichi's **abstract parallel language**.

- If the output is "slightly different," it is almost always:
  - **Propagation order** (Forward vs Backward scans).
  - **Rounding bias** (`+0.5` mismatch).
  - **Random distribution** (Uniform vs biased truncation).

---

## 10. The "Invisible" Bottlenecks (Performance Discoveries)

Speed is the goal, but intuition often fails on GPU.

- **Dynamic Memory Allocation**: Never use `.to_numpy()` or `.from_numpy()` inside the `run_level` loop. It triggers a CPU-GPU sync that destroys performance.
- **Python-Side Branching**: Minimize `if` statements in the main Python loop that call different kernels. Each call incurs a launch overhead.
- **Vectorized vs Parallel**: If a task can be done in 1 line of PyTorch (like `tensor.abs().max()`), do it there. Don't write a Taichi kernel for a reduction unless the data is already on the Taichi device and you want to avoid a sync.

---

## 11. Agent-to-Agent Debugging Checklist

Before you tell the user "it matches," run this checklist:

- [ ] **Coordinate Parity**: Does `(0,0)` in your NNF point to the center of the patch or the top-left? (EBSynth uses centers).
- [ ] **Bounds Parity**: Are you using `clamp` or `in-bounds` checks? (CUDA uses both; Taichi should match).
- [ ] **Weighting Parity**: Are your guide weights applied *before* or *after* the square? (SSD is `(w * diff)**2`? No, usually `w * (diff**2)`).
- [ ] **Iteration Parity**: Does `num_iters=2` mean 2 forward passes, or 1 forward + 1 backward? (Standard PM is Forward-Backward pairs).
- [ ] **Rounding Parity**: Cast to `u8` only at the very end of reconstruction. Keep all intermediate accumulation in `f32`.

---

## 12. Advanced Troubleshooting Table

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| `SPIR-V error` | Mapped to `f64` on Metal hardware. | Check for `np.float64` inputs. Force `float32`. |
| `Dim1 accessed with Dim3` | Kernel signature mismatch. | Pass `(1,1,1)` dummy tensors for optional 3D args. |
| `Atomic add slow` | Buffer contention. | Avoid `ti.atomic_add` on global fields in high-traffic loops. |
| `MSL compilation error` | 64-bit atomics on Metal. | Switch to 31-bit packing with `ti.i32`. |
| `Tiling Artifacts` | Coordinate truncation. | Increase bit-depth for coordinates (e.g. 10 bits for 1024px). |
| `Kernel bias toward 0` | `int(ti.random())` truncation. | Use `ti.floor()` for symmetric distribution. |
| `Random results` | Seed non-determinism. | Use `ti.init(random_seed=42)` for verification. |

---

## 13. Final Secret: The "Sequential-Parallel" Trick

Taichi can't easily propagate information across the grid *during* a parallel loop (like a Scan operation).
**Discovery**: For Forward/Backward passes in PatchMatch, we simulate the sequentiality by running the passes in the correct order in **Python**, while keeping the **pixel-local** propagation (checking Top/Left neighbors) inside the parallel Taichi kernel. This hybrid approach is what gives the 90x speedup while maintaining the "scanline" logic of PatchMatch.

> *End of Reference*
