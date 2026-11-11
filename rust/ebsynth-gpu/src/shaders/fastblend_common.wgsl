// FastBlend pyramid patch matching (WGSL port of FastBlend/src/backends/patch_matcher_core.py
// and the remap / patch-error kernels of FastBlend/src/backends/taichi_kernels.py).
//
// Images are float32 (B, H, W, C) row-major with C <= 4. Padded images carry a reflect
// border of `pad` pixels. NNFs are int32 (B, h, w, 2) holding (row, col) source coordinates.
// Kernels run over (col, row, batch) = global_invocation_id.

struct Params {
    h: i32, w: i32, c: i32, pad: i32,          // level size, channels, padding
    full_h: i32, full_w: i32, batch: i32, style_mode: u32,
    prev_h: i32, prev_w: i32, _p0: i32, _p1: i32,
    guide_weight: f32, _p2: f32, _p3: f32, _p4: f32,
}

// Per-dispatch arguments: patch radius, candidate kind, random range / track offset, seed.
struct Step { r: i32, kind: u32, arg: i32, seed: u32 }

const STYLE_PLAIN: u32 = 0u;
const STYLE_MEAN: u32 = 1u;
const STYLE_PAIRWISE: u32 = 2u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Step;

fn padded_index(b: i32, y: i32, x: i32) -> i32 {
    let hp = P.h + 2 * P.pad;
    let wp = P.w + 2 * P.pad;
    return ((b * hp + y) * wp + x) * P.c;
}

fn nnf_index(b: i32, y: i32, x: i32) -> i32 {
    return (b * P.h + y) * P.w + x;
}

fn in_level(g: vec3<u32>) -> bool {
    return i32(g.x) < P.w && i32(g.y) < P.h && i32(g.z) < P.batch;
}

fn pcg(state: ptr<function, u32>) -> u32 {
    let s = *state * 747796405u + 2891336453u;
    *state = s;
    let v = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (v >> 22u) ^ v;
}

fn rng(b: i32, y: i32, x: i32) -> u32 {
    var s = (u32(nnf_index(b, y, x)) * 2654435761u) ^ (S.seed * 0x9E3779B9u);
    _ = pcg(&s);
    return s;
}
