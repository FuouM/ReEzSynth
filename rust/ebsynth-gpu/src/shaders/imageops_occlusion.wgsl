// Flow occlusion masks (ezsynth/utils/occlusion.py): forward-backward consistency plus
// coverage holes of the rounded forward splat, dilated like cv2.dilate.

@group(0) @binding(1) var<storage, read_write> primary: array<f32>;
@group(0) @binding(2) var<storage, read_write> secondary: array<f32>;
@group(0) @binding(3) var<storage, read_write> m0: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> m1: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> m2: array<atomic<u32>>;
// (dx, dy) pairs of the dilation structuring element; p.n pairs.
@group(0) @binding(6) var<storage, read_write> offsets: array<i32>;

fn secondary_tap(y: i32, x: i32, ch: u32) -> f32 {
    if (y < 0 || y >= i32(p.h) || x < 0 || x >= i32(p.w)) {
        return 0.0;
    }
    return secondary[(u32(y) * p.w + u32(x)) * 2u + ch];
}

// cv2.remap(secondary, INTER_LINEAR, BORDER_CONSTANT 0) at (mx, my).
fn sample_secondary(mx: f32, my: f32) -> vec2<f32> {
    let fx = remap_fixed(mx);
    let fy = remap_fixed(my);
    let x0 = fx >> 5u;
    let y0 = fy >> 5u;
    let tx = f32(fx & 31) * 0.03125;
    let ty = f32(fy & 31) * 0.03125;
    let w00 = (1.0 - ty) * (1.0 - tx);
    let w01 = (1.0 - ty) * tx;
    let w10 = ty * (1.0 - tx);
    let w11 = ty * tx;
    var out = vec2<f32>(0.0);
    for (var ch = 0u; ch < 2u; ch++) {
        out[ch] = secondary_tap(y0, x0, ch) * w00 + secondary_tap(y0, x0 + 1, ch) * w01
            + secondary_tap(y0 + 1, x0, ch) * w10 + secondary_tap(y0 + 1, x0 + 1, ch) * w11;
    }
    return out;
}

// m0 = 1 where primary then secondary flow do not return home (f.x alpha, f.y beta)
// or the primary flow leaves the frame.
@compute @workgroup_size(16, 16)
fn consistency(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let pf = vec2<f32>(primary[2u * i], primary[2u * i + 1u]);
    let tx = f32(id.x) + pf.x;
    let ty = f32(id.y) + pf.y;
    let sf = sample_secondary(tx, ty);
    let r = pf + sf;
    let residual = r.x * r.x + r.y * r.y;
    let mag = (pf.x * pf.x + pf.y * pf.y) + (sf.x * sf.x + sf.y * sf.y);
    let outside = tx < 0.0 || tx > f32(p.w - 1u) || ty < 0.0 || ty > f32(p.h - 1u);
    atomicStore(&m0[i], select(0u, 1u, residual > p.f.x * mag + p.f.y || outside));
}

// m1 = dilation of m0 by the structuring element in `offsets`.
@compute @workgroup_size(16, 16)
fn dilate_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    var v = 0u;
    for (var k = 0u; k < p.n; k++) {
        let x = i32(id.x) + offsets[2u * k];
        let y = i32(id.y) + offsets[2u * k + 1u];
        if (x >= 0 && x < i32(p.w) && y >= 0 && y < i32(p.h)) {
            v = max(v, atomicLoad(&m0[u32(y) * p.w + u32(x)]));
        }
    }
    atomicStore(&m1[id.y * p.w + id.x], v);
}

// m2 = 1 at every pixel hit by a source pixel moved by the rounded secondary flow.
@compute @workgroup_size(16, 16)
fn scatter_cover(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let x = round(f32(id.x) + secondary[2u * i]);
    let y = round(f32(id.y) + secondary[2u * i + 1u]);
    if (x >= 0.0 && x < f32(p.w) && y >= 0.0 && y < f32(p.h)) {
        atomicStore(&m2[u32(y) * p.w + u32(x)], 1u);
    }
}

// m0 = 3x3 rectangular dilation of the coverage map m2.
@compute @workgroup_size(16, 16)
fn dilate_cover(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    var v = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let x = i32(id.x) + dx;
            let y = i32(id.y) + dy;
            if (x >= 0 && x < i32(p.w) && y >= 0 && y < i32(p.h)) {
                v = max(v, atomicLoad(&m2[u32(y) * p.w + u32(x)]));
            }
        }
    }
    atomicStore(&m0[id.y * p.w + id.x], v);
}

// m2 = 255 where the dilated consistency mask is set or (p.flag 1) coverage is missing.
@compute @workgroup_size(16, 16)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let hole = p.flag == 1u && atomicLoad(&m0[i]) == 0u;
    atomicStore(&m2[i], select(0u, 255u, atomicLoad(&m1[i]) != 0u || hole));
}
