# Synthesis stages by backend

Where each stage of one pyramid level lives in every backend. Paths are relative to
`ezsynth/kernels/synth/`; `wgsl` means `rust/ebsynth-gpu/src/shaders/ebsynth.wgsl`.
`-` means the backend does not implement that option.

| Stage | `cuda` (`cuda_src/`) | `cuda` on CPU (`cuda_src/cpu/`) | `torch` (`torch.py`, `torch_ops/`) | `taichi` (`taichi_kernels.py`) | `wgpu` (`wgsl`) |
|---|---|---|---|---|---|
| Level driver | `dispatch.cu` `run_level_cuda` | `dispatch_cpu.cpp` `ebsynth_cpu_run_level` | `Backend.run_level` | `taichi.py` `Backend.run_level` | `rust/ebsynth-gpu/src/ebsynth.rs` `plan_level` |
| RNG state | `utils.cu` `init_rand_states_kernel` | `dispatch_cpu.cpp` `init_rand_states_cpu` | `Backend._make_random_generator` | `ti.random` in the kernels | `pcg`, `rng_seed` |
| Uniformity (omega) map | `omega_ops.cu` `populate_initial_omega_kernel` | `omega_ops_cpu.cpp` `populate_initial_omega_cpu` | `omega_ops.populate_omega_map` | `populate_omega` | `init_omega` |
| Integral image (NCC) | `integral_image.cu` `compute_integral_image_cuda` | `integral_image_cpu.cpp` `compute_integral_image_cpu` | `patch_ops.compute_patch_stats` | `compute_integral_image` | `sat_rows`, `sat_cols` |
| Patch cost (SSD / NCC / SAD) | `cost_functions.cu` (SSD, NCC) | `cost_functions_cpu.cpp` (SSD, NCC) | `patch_ops.compute_patch_ssd_vectorized`, `compute_patch_ssd_style_guide_fused`, `compute_patch_ncc_vectorized`; bilateral or `n_size_step > 1`: `compute_patch_cost_sampled`; `patchmatch_ops.grid_patch_errors` | `compute_patch_ssd`, `compute_patch_ncc`, `patch_cost` | `cost_ssd`, `cost_ncc`, `patch_cost` |
| Initial error map | `patchmatch.cu` `compute_initial_error_kernel` | `patchmatch_cpu.cpp` `compute_initial_error_cpu` | `Backend._current_errors` | `compute_error_map_kernel` | `error_map` |
| Try a candidate (cost + omega change) | `patchmatch.cu` | `patchmatch_cpu.cpp` `try_patch_cpu` | `patchmatch_ops.try_patch_batch` | `get_omega`, `patch_omega_atomic_add` | `omega_term`, `omega_add` |
| Propagation | `patchmatch.cu` `propagation_step_kernel` | `patchmatch_cpu.cpp` `propagation_step_cpu` | `patchmatch_ops.propagation_step` (`Backend._propagate`) | `patchmatch_step_kernel` | `patchmatch_step` |
| Random search | `patchmatch.cu` `random_search_step_kernel` | `patchmatch_cpu.cpp` `random_search_step_cpu` | `patchmatch_ops.random_search_step` (`Backend._random_search`) | `random_search_kernel` | `random_search` |
| Jump flooding (`search_mode="jfa"`) | - | - | `jfa_ops.jfa_search`, `jfa_step` (`Backend._jfa`) | `jfa_step_kernel` | `jfa_step` |
| Multi-candidate (`search_mode="multicand"`) | - | - | `multicand_ops.initial_candidate_set`, `propagation_step_multicand`, `refresh_and_promote` | `multicand_refresh_kernel`, `multicand_promote_kernel`, `multicand_propagation_kernel`, `multicand_random_search_kernel` | `init_alt`, `multicand_refresh`, `multicand_promote`, `multicand_step`, `multicand_random_search` |
| Vote, plain | `voting.cu` `krnlVotePlain` | `voting_cpu.cpp` `krnlVotePlain_cpu` | `voting_ops.vote_plain` (bilateral: `_vote_bilateral`) | `vote_plain_kernel` | `vote` |
| Vote, weighted | `voting.cu` `krnlVoteWeighted` | `voting_cpu.cpp` `krnlVoteWeighted_cpu` | `voting_ops.vote_weighted` (bilateral: `_vote_bilateral`) | `vote_weighted_kernel`, `vote_weighted_multicand_kernel`; bilateral: `voting_kernel` | `vote`, `vote_bilateral`, `vote_multicand` |
| Stop mask (`stop_threshold`) | `mask_ops.cu` `eval_mask_kernel`, `dilate_mask_kernel` | `mask_ops_cpu.cpp` `eval_mask_cpu`, `dilate_mask_cpu` | `mask_ops.evaluate_mask`, `dilate_mask` | `eval_mask_kernel`, `dilate_mask_kernel` | `eval_mask`, `dilate_mask` |
