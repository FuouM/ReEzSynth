// Shared by the image-op modules: one uniform block plus six storage bindings that each
// module types for its own kernels.

struct Params {
    h: u32,
    w: u32,
    c: u32,
    n: u32,
    h2: u32,
    w2: u32,
    mode: u32,
    flag: u32,
    f: vec4<f32>,
    s: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> p: Params;

// cv2.remap fixed-point coordinate: 1/32 pixel steps, rounded half to even like cvRound.
fn remap_fixed(v: f32) -> i32 {
    return i32(clamp(round(v * 32.0), -1073741824.0, 1073741760.0));
}

// cv2 BORDER_REFLECT index (fedcba|abcdefgh|hgfedcb).
fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) {
        return 0;
    }
    let period = 2 * len;
    var q = i % period;
    if (q < 0) {
        q += period;
    }
    if (q >= len) {
        q = period - 1 - q;
    }
    return q;
}

fn finite_or_zero(v: f32) -> f32 {
    if (v != v || v > 1.0e20 || v < -1.0e20) {
        return 0.0;
    }
    return v;
}

fn floor_index(v: f32) -> i32 {
    return i32(clamp(floor(v), -1.0e9, 1.0e9));
}
