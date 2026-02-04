import platform

import numpy as np
import taichi as ti
import torch


# Initialize Taichi
def get_ti_arch():
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return ti.metal
    elif torch.cuda.is_available():
        return ti.cuda
    return ti.cpu


_ti_initialized = False


def ensure_ti_init():
    global _ti_initialized
    if not _ti_initialized:
        arch = get_ti_arch()
        ti.init(arch=arch, log_level=ti.INFO, random_seed=42)
        print(f"[Taichi FaceBlit] Initialized with arch: {arch}")
        _ti_initialized = True


@ti.data_oriented
class FaceBlitTaichiBackend:
    def __init__(self):
        ensure_ti_init()
        self.device = "cpu"
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

    # --- LUT COMPUTATION KERNELS ---

    @ti.kernel
    def compute_lut_search_packed_kernel(
        self,
        style_pos_r: ti.types.ndarray(),
        style_pos_g: ti.types.ndarray(),
        style_app: ti.types.ndarray(),
        packed_min: ti.types.ndarray(),
        lambda_pos: int,
        search_radius: int,
        w_s: int,
        h_s: int,
        x_norm: float,
        y_norm: float,
    ):
        # Parallelize over target_x and target_y
        for tr, tg in ti.ndrange(256, 256):
            # tr is target Red (X guide), tg is target Green (Y guide)
            seed_col = int(tr * x_norm)
            seed_row = int(tg * y_norm)

            c_start = ti.max(0, seed_col - search_radius)
            c_end = ti.min(w_s, seed_col + search_radius + 1)
            r_start = ti.max(0, seed_row - search_radius)
            r_end = ti.min(h_s, seed_row + search_radius + 1)

            for sx in range(c_start, c_end):
                for sy in range(r_start, r_end):
                    app_val = style_app[sy, sx]
                    # Max error is 5100. Scale by 4 to fit in 11 bits (max 2047).
                    err = (
                        ti.abs(int(style_pos_r[sy, sx]) - tr)
                        + ti.abs(int(style_pos_g[sy, sx]) - tg)
                    ) * lambda_pos
                    clamped_err = ti.min(err // 4, 2047)

                    # Pack: (cost << 20) | (sx << 10) | sy
                    # Uses 31 bits total to stay within positive int32 range
                    new_val = (
                        (ti.i32(clamped_err) << 20) | (ti.i32(sx) << 10) | ti.i32(sy)
                    )
                    ti.atomic_min(packed_min[tr, tg, app_val], new_val)

    @ti.kernel
    def compute_lut_smooth_kernel(
        self,
        packed_min: ti.types.ndarray(),
        lambda_app: int,
        seed_x: ti.types.ndarray(),
        seed_y: ti.types.ndarray(),
    ):
        # Linear smoothing over app_val (z) for each (tx, ty)
        k_inf_cost = ti.i32(2047)
        # Scaled lambda_app
        l_app_scaled = ti.max(1, lambda_app // 4)

        for tr, tg in ti.ndrange(256, 256):
            s_x = ti.i32(seed_x[tr])
            s_y = ti.i32(seed_y[tg])
            seed_coords_packed = (s_x << 10) | s_y

            # Forward pass
            prev_cost = k_inf_cost
            prev_coords_packed = seed_coords_packed
            for z in range(256):
                curr_packed = packed_min[tr, tg, z]
                curr_cost = curr_packed >> 20

                prop_cost = prev_cost + l_app_scaled
                if prop_cost < curr_cost:
                    packed_min[tr, tg, z] = (
                        ti.i32(ti.min(prop_cost, 2047)) << 20
                    ) | prev_coords_packed
                    prev_cost = prop_cost
                else:
                    prev_cost = curr_cost
                    prev_coords_packed = curr_packed & 0xFFFFF  # 20 bits for coords

            # Backward pass
            prev_cost = k_inf_cost
            prev_coords_packed = seed_coords_packed
            for z_inv in range(256):
                z = 255 - z_inv
                curr_packed = packed_min[tr, tg, z]
                curr_cost = curr_packed >> 20

                prop_cost = prev_cost + l_app_scaled
                if prop_cost < curr_cost:
                    packed_min[tr, tg, z] = (
                        ti.i32(ti.min(prop_cost, 2047)) << 20
                    ) | prev_coords_packed
                    prev_cost = prop_cost
                else:
                    prev_cost = curr_cost
                    prev_coords_packed = curr_packed & 0xFFFFF

            # Final hole filling
            for z in range(256):
                if (packed_min[tr, tg, z] >> 20) >= k_inf_cost:
                    packed_min[tr, tg, z] = (k_inf_cost << 20) | seed_coords_packed

    # --- STYLIZATION KERNELS ---

    @ti.kernel
    def lookup_nnf_kernel(
        self,
        target_pos_r: ti.types.ndarray(),
        target_pos_g: ti.types.ndarray(),
        target_app: ti.types.ndarray(),
        look_up_cube_packed: ti.types.ndarray(),
        nnf: ti.types.ndarray(),
    ):
        for ty, tx in ti.ndrange(target_app.shape[0], target_app.shape[1]):
            tr = target_pos_r[ty, tx]
            tg = target_pos_g[ty, tx]
            ta = target_app[ty, tx]

            # Using Red channel as X and Green channel as Y to index the LUT
            packed = look_up_cube_packed[tr, tg, ta]
            nnf[ty, tx, 1] = int((packed >> 10) & 0x3FF)  # sx (10 bits)
            nnf[ty, tx, 0] = int(packed & 0x3FF)  # sy (10 bits)

    @ti.kernel
    def denoise_nnf_kernel(
        self, nnf: ti.types.ndarray(), denoised_nnf: ti.types.ndarray(), patch_size: int
    ):
        h, w = nnf.shape[0], nnf.shape[1]
        r = patch_size // 2
        for ty, tx in ti.ndrange(h, w):
            # Mode filter on offsets to maintain edge sharpness
            # Pack offsets: ((sy-ty) << 16) | (sx-tx)
            # We use a simple histogram/voting approach for mode
            best_y_off = 0
            best_x_off = 0
            # For small patches (3x3), we can use a small fixed-size search
            # But Taichi doesn't easily support dynamic sorting in kernels.
            # FaceBlit C++ uses a hash map or sorting.
            # Simplified: just pick the majority offset in patch.

            # Since we can't easily have a dynamic histogram,
            # we'll use a nested loop to find the most frequent offset.
            max_count = 0
            for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
                ny, nx = (
                    ti.max(0, ti.min(h - 1, ty + py)),
                    ti.max(0, ti.min(w - 1, tx + px)),
                )
                curr_y_off = nnf[ny, nx, 0] - ny
                curr_x_off = nnf[ny, nx, 1] - nx

                count = 0
                for py2, px2 in ti.ndrange((-r, r + 1), (-r, r + 1)):
                    ny2, nx2 = (
                        ti.max(0, ti.min(h - 1, ty + py2)),
                        ti.max(0, ti.min(w - 1, tx + px2)),
                    )
                    if (nnf[ny2, nx2, 0] - ny2 == curr_y_off) and (
                        nnf[ny2, nx2, 1] - nx2 == curr_x_off
                    ):
                        count += 1

                if count > max_count:
                    max_count = count
                    best_y_off = curr_y_off
                    best_x_off = curr_x_off

            denoised_nnf[ty, tx, 0] = best_y_off + ty
            denoised_nnf[ty, tx, 1] = best_x_off + tx

    @ti.kernel
    def voting_kernel(
        self,
        output_image: ti.types.ndarray(),
        source_style: ti.types.ndarray(),
        nnf: ti.types.ndarray(),
        patch_size: int,
    ):
        th, tw = output_image.shape[0], output_image.shape[1]
        sh, sw = source_style.shape[0], source_style.shape[1]
        r = patch_size // 2
        for ty, tx in ti.ndrange(th, tw):
            sum_b = 0.0
            sum_g = 0.0
            sum_r = 0.0
            count = 0.0
            for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
                ny, nx = ty + py, tx + px
                if 0 <= ny < th and 0 <= nx < tw:
                    sy = nnf[ny, nx, 0] - py
                    sx = nnf[ny, nx, 1] - px
                    if 0 <= sy < sh and 0 <= sx < sw:
                        sum_b += float(source_style[sy, sx, 0])
                        sum_g += float(source_style[sy, sx, 1])
                        sum_r += float(source_style[sy, sx, 2])
                        count += 1.0
            if count > 0:
                output_image[ty, tx, 0] = ti.u8(ti.max(0, ti.min(255, sum_b / count)))
                output_image[ty, tx, 1] = ti.u8(ti.max(0, ti.min(255, sum_g / count)))
                output_image[ty, tx, 2] = ti.u8(ti.max(0, ti.min(255, sum_r / count)))
            else:
                # Fallback to nearest neighbor if voting failed
                sy = nnf[ty, tx, 0]
                sx = nnf[ty, tx, 1]
                output_image[ty, tx, 0] = source_style[sy, sx, 0]
                output_image[ty, tx, 1] = source_style[sy, sx, 1]
                output_image[ty, tx, 2] = source_style[sy, sx, 2]

    # --- WRAPPERS ---

    def compute_lut(
        self,
        style_pos_guide,
        style_app_guide,
        lambda_pos=10,
        lambda_app=2,
        search_radius=30,
    ):
        h_s, w_s = style_app_guide.shape[:2]
        # Using 31-bit packing: (cost << 20) | (sx << 10) | sy
        # Init with large cost (2047 << 20)
        k_inf_packed = 2047 << 20
        packed_min = torch.full(
            (256, 256, 256), k_inf_packed, dtype=torch.int32, device=self.device
        )

        style_pos_r = style_pos_guide[..., 2].to(self.device).contiguous()
        style_pos_g = style_pos_guide[..., 1].to(self.device).contiguous()
        style_app = style_app_guide.to(self.device).contiguous()

        x_norm = w_s / 256.0
        y_norm = h_s / 256.0

        self.compute_lut_search_packed_kernel(
            style_pos_r,
            style_pos_g,
            style_app,
            packed_min,
            lambda_pos,
            search_radius,
            w_s,
            h_s,
            x_norm,
            y_norm,
        )

        seed_x = (torch.arange(256).float() * x_norm).short().to(self.device)
        seed_y = (torch.arange(256).float() * y_norm).short().to(self.device)
        self.compute_lut_smooth_kernel(packed_min, lambda_app, seed_x, seed_y)

        return packed_min

    def stylize(
        self,
        style_image,
        target_pos_guide,
        target_app_guide,
        look_up_cube_packed,
        patch_size=3,
    ):
        h_t, w_t = target_app_guide.shape[:2]
        nnf = torch.zeros((h_t, w_t, 2), dtype=torch.int32, device=self.device)
        output = torch.zeros((h_t, w_t, 3), dtype=torch.uint8, device=self.device)

        tp_r = target_pos_guide[..., 2].to(self.device).contiguous()
        tp_g = target_pos_guide[..., 1].to(self.device).contiguous()
        ta = target_app_guide.to(self.device).contiguous()
        style_image_ti = style_image.to(self.device).contiguous()

        self.lookup_nnf_kernel(tp_r, tp_g, ta, look_up_cube_packed.to(self.device), nnf)

        if patch_size > 1:
            denoised_nnf = torch.zeros_like(nnf)
            self.denoise_nnf_kernel(nnf, denoised_nnf, patch_size)
            nnf = denoised_nnf

        self.voting_kernel(output, style_image_ti, nnf, patch_size)
        return output
