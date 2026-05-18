"""FastBlend Taichi kernels (module-level, no class).

Same layout as ``ezsynth/engines/backends/taichi_kernels.py``: ``ti.types.ndarray()``
arguments come first, then scalar ``int`` / ``float``. Remap / patch SSD mirror
``fastblend_extension/kernels.cu``.
"""

import taichi as ti


@ti.kernel
def remap_kernel(
    source_style: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
    target_style: ti.types.ndarray(),
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
):
    r = (patch_size - 1) // 2
    for b, x, y in ti.ndrange(source_style.shape[0], height, width):
        num = 0.0

        min_px = -x if x < r else -r
        max_px = height - 1 - x if x + r > height - 1 else r
        min_py = -y if y < r else -r
        max_py = width - 1 - y if y + r > width - 1 else r

        for px, py in ti.ndrange((min_px, max_px + 1), (min_py, max_py + 1)):
            nx, ny = x + px, y + py

            sy_match = nnf[b, nx, ny, 0]
            sx_match = nnf[b, nx, ny, 1]

            sy = sy_match - px
            sx = sx_match - py

            if 0 <= sy < height and 0 <= sx < width:
                num += 1.0
                for c in range(channel):
                    target_style[b, x + pad_size, y + pad_size, c] += source_style[
                        b, sy + pad_size, sx + pad_size, c
                    ]

        if num > 0:
            for c in range(channel):
                target_style[b, x + pad_size, y + pad_size, c] /= num


@ti.kernel
def patch_error_kernel(
    source: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
    target: ti.types.ndarray(),
    error: ti.types.ndarray(),
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
):
    r = (patch_size - 1) // 2
    for b, y, x in ti.ndrange(source.shape[0], height, width):
        sy_match = nnf[b, y, x, 0]
        sx_match = nnf[b, y, x, 1]

        e = 0.0
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            typ, txp = y + pad_size + py, x + pad_size + px
            syp, sxp = sy_match + pad_size + py, sx_match + pad_size + px

            for c in range(channel):
                diff = target[b, typ, txp, c] - source[b, syp, sxp, c]
                e += diff * diff
        error[b, y, x] = e


@ti.kernel
def pairwise_patch_error_kernel(
    source_a: ti.types.ndarray(),
    nnf_a: ti.types.ndarray(),
    source_b: ti.types.ndarray(),
    nnf_b: ti.types.ndarray(),
    error: ti.types.ndarray(),
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
):
    r = (patch_size - 1) // 2
    for b, y, x in ti.ndrange(source_a.shape[0], height, width):
        sy_a = nnf_a[b, y, x, 0]
        sx_a = nnf_a[b, y, x, 1]
        sy_b = nnf_b[b, y, x, 0]
        sx_b = nnf_b[b, y, x, 1]

        e = 0.0
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            syp_a, sxp_a = sy_a + pad_size + py, sx_a + pad_size + px
            syp_b, sxp_b = sy_b + pad_size + py, sx_b + pad_size + px

            for c in range(channel):
                diff = source_a[b, syp_a, sxp_a, c] - source_b[b, syp_b, sxp_b, c]
                e += diff * diff
        error[b, y, x] = e
