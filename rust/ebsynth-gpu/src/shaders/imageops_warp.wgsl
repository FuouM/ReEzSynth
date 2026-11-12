// Backward warping (cv2.remap) and the pull-push / normalize passes of forward splatting
// (Taichi kernels in ezsynth/engines/backends/taichi_ops.py).

@group(0) @binding(1) var<storage, read_write> src: array<f32>;
// warp: flow (h, w, 2); pull / push / normalize: weight of `src`.
@group(0) @binding(2) var<storage, read_write> aux: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_w: array<f32>;

// Pixel of `src` (p.h, p.w, p.c); p.mode 0: BORDER_CONSTANT (0), 1: BORDER_REFLECT.
fn tap(y: i32, x: i32, ch: u32) -> f32 {
    let h = i32(p.h);
    let w = i32(p.w);
    var yy = y;
    var xx = x;
    if (p.mode == 1u) {
        yy = reflect_index(y, h);
        xx = reflect_index(x, w);
    } else if (y < 0 || y >= h || x < 0 || x >= w) {
        return 0.0;
    }
    return src[(u32(yy) * p.w + u32(xx)) * p.c + ch];
}

// dst(y, x) = src(grid + flow); p.flag 0: INTER_LINEAR, 1: INTER_NEAREST. Bilinear
// coordinates are exact floats, or OpenCV's 1/32-pixel fixed point when p.n == 1.
@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let mx = f32(id.x) + aux[2u * i];
    let my = f32(id.y) + aux[2u * i + 1u];
    if (p.flag == 1u) {
        let sx = i32(clamp(round(mx), -1073741824.0, 1073741760.0));
        let sy = i32(clamp(round(my), -1073741824.0, 1073741760.0));
        for (var ch = 0u; ch < p.c; ch++) {
            dst[i * p.c + ch] = tap(sy, sx, ch);
        }
        return;
    }
    var x0: i32;
    var y0: i32;
    var tx: f32;
    var ty: f32;
    if (p.n == 1u) {
        let fx = remap_fixed(mx);
        let fy = remap_fixed(my);
        x0 = fx >> 5u;
        y0 = fy >> 5u;
        tx = f32(fx & 31) * 0.03125;
        ty = f32(fy & 31) * 0.03125;
    } else {
        x0 = floor_index(mx);
        y0 = floor_index(my);
        tx = mx - floor(mx);
        ty = my - floor(my);
    }
    let w00 = (1.0 - ty) * (1.0 - tx);
    let w01 = (1.0 - ty) * tx;
    let w10 = ty * (1.0 - tx);
    let w11 = ty * tx;
    for (var ch = 0u; ch < p.c; ch++) {
        dst[i * p.c + ch] = tap(y0, x0, ch) * w00 + tap(y0, x0 + 1, ch) * w01
            + tap(y0 + 1, x0, ch) * w10 + tap(y0 + 1, x0 + 1, ch) * w11;
    }
}

// Pull: 5x5 Gaussian-weighted sums of level (p.h, p.w) into level (p.h2, p.w2).
@compute @workgroup_size(16, 16)
fn pull(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w2 || id.y >= p.h2) {
        return;
    }
    let i = i32(id.y);
    let j = i32(id.x);
    var sum_w = 0.0;
    for (var ch = 0u; ch < p.c; ch++) {
        var sum_c = 0.0;
        for (var di = -2; di <= 2; di++) {
            for (var dj = -2; dj <= 2; dj++) {
                let si = i * 2 + di;
                let sj = j * 2 + dj;
                if (si >= 0 && si < i32(p.h) && sj >= 0 && sj < i32(p.w)) {
                    let gw = exp(-f32(di * di + dj * dj) / 2.0);
                    let s = u32(si) * p.w + u32(sj);
                    sum_c += finite_or_zero(src[s * p.c + ch]) * gw;
                    if (ch == 0u) {
                        sum_w += finite_or_zero(aux[s]) * gw;
                    }
                }
            }
        }
        dst[(id.y * p.w2 + id.x) * p.c + ch] = sum_c;
    }
    dst_w[id.y * p.w2 + id.x] = finite_or_zero(sum_w);
}

// Push: fill weak pixels of the finer level (p.h2, p.w2) from the coarser level (p.h, p.w).
@compute @workgroup_size(16, 16)
fn push(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w2 || id.y >= p.h2) {
        return;
    }
    let fi = id.y * p.w2 + id.x;
    let dw = clamp(finite_or_zero(dst_w[fi]), 0.0, 1.0);
    if (dw >= 0.95) {
        return;
    }
    // Odd sizes: the last fine row/column maps past the coarse level; use its edge.
    let ci = min(id.y / 2u, p.h - 1u) * p.w + min(id.x / 2u, p.w - 1u);
    let sw = finite_or_zero(aux[ci]);
    if (sw <= 1e-4) {
        return;
    }
    let alpha = clamp(1.0 - dw, 0.0, 1.0);
    for (var ch = 0u; ch < p.c; ch++) {
        let coarse = finite_or_zero(src[ci * p.c + ch] / sw);
        let current = finite_or_zero(dst[fi * p.c + ch] / max(dw, 1e-6));
        dst[fi * p.c + ch] = dw * current + alpha * coarse;
    }
    dst_w[fi] = 1.0;
}

// dst = src / weight where the weight is usable, else 0.
@compute @workgroup_size(16, 16)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let weight = finite_or_zero(aux[i]);
    let usable = weight > 1e-4 && weight < 1.0e20;
    for (var ch = 0u; ch < p.c; ch++) {
        var v = 0.0;
        if (usable) {
            v = finite_or_zero(src[i * p.c + ch] / weight);
        }
        dst[i * p.c + ch] = v;
    }
}
