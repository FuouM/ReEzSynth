"""FaceBlit kernels (look-up-cube patch matching for faces), one module per engine.

``FaceBlit/src/engine.py`` picks the engine per stage (``backend`` torch/taichi,
``dfs_mode`` numba/python); ``FaceBlit/src/commons.py`` holds the guide, warp and mask
helpers that have a single implementation.

| Stage                        | ``numpy``                                   | ``numba``                                        | ``torch``                         | ``taichi``                                   |
|------------------------------|---------------------------------------------|--------------------------------------------------|-----------------------------------|----------------------------------------------|
| Style look-up cube (LUT)     | ``compute_look_up_cube``                    | -                                                | ``compute_look_up_cube_optimized`` | ``compute_lut_taichi``                      |
| NNF from the LUT             | ``_lookup_coords_from_guides``              | -                                                | in ``style_blit_voting_torch``    | ``lookup_nnf_kernel``                        |
| DFS seed-grow NNF            | ``dfs_seed_grow_voting_numpy`` (driven by ``initialize_nnf`` in engine.py) | ``initialize_nnf_dfs_voting_numba``, ``dfs_seed_grow_voting_numba`` | - | ``initialize_nnf_dfs_kernel``                |
| NNF denoise (mode filter)    | ``denoise_nnf_python``                      | ``denoise_nnf_numba``                            | ``_denoise_nnf_torch``            | ``denoise_nnf_kernel``                       |
| Voting                       | ``voting_on_rgb``                           | -                                                | ``_voting_on_rgb_torch``          | ``voting_kernel``                            |
| Whole stylization            | ``style_blit``                              | -                                                | ``style_blit_voting_torch``       | ``stylize_blit_taichi``                      |

Unlike ``synth`` and ``fastblend`` these kernels share no code with the other families.
"""
