// Histogram blending (ezsynth/utils/blend_logic.py hist_blender, OpenCV path): blend two
// BGR frames in normalized Lab space, then match the statistics of the per-pixel
// lower-error frame. Lab values use OpenCV's 8-bit encoding (L * 255 / 100, a + 128,
// b + 128). Per-workgroup partial sums go to `part`; the host reduces them.

@group(0) @binding(1) var<storage, read_write> img_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> img_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;
// hist_lab: packed Lab of a and b (2 words per pixel); hist_final: packed BGR output.
@group(0) @binding(4) var<storage, read_write> lab: array<u32>;
@group(0) @binding(5) var<storage, read_write> ab: array<f32>;
@group(0) @binding(6) var<storage, read_write> part: array<u32>;

var<workgroup> acc: array<atomic<u32>, 18>;
var<workgroup> red_s: array<vec3<f32>, 256>;
var<workgroup> red_q: array<vec3<f32>, 256>;

fn byte_a(i: u32) -> u32 {
    return (img_a[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn byte_b(i: u32) -> u32 {
    return (img_b[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn byte_mask(i: u32) -> u32 {
    return (mask[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn pack3(v: vec3<u32>) -> u32 {
    return v.x | (v.y << 8u) | (v.z << 16u);
}

fn unpack3(v: u32) -> vec3<f32> {
    return vec3<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u));
}

fn srgb_to_linear(x: f32) -> f32 {
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb(x: f32) -> f32 {
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn lab_f(t: f32) -> f32 {
    if (t > 0.008856) {
        return pow(t, 1.0 / 3.0);
    }
    return 7.787 * t + 16.0 / 116.0;
}

fn lab_f_inv(t: f32) -> f32 {
    if (t <= 0.206893) {
        return (t - 16.0 / 116.0) / 7.787;
    }
    return t * t * t;
}

// 8-bit BGR -> 8-bit Lab (OpenCV COLOR_BGR2Lab, D65, sRGB).
fn bgr_to_lab8(bgr: vec3<f32>) -> vec3<u32> {
    let r = srgb_to_linear(bgr.z / 255.0);
    let g = srgb_to_linear(bgr.y / 255.0);
    let b = srgb_to_linear(bgr.x / 255.0);
    let x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456;
    let y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    let z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);
    let l = (116.0 * fy - 16.0) * 255.0 / 100.0;
    let v = vec3<f32>(l, 500.0 * (fx - fy) + 128.0, 200.0 * (fy - fz) + 128.0);
    return vec3<u32>(clamp(round(v), vec3<f32>(0.0), vec3<f32>(255.0)));
}

// 8-bit Lab -> 8-bit BGR (OpenCV COLOR_Lab2BGR).
fn lab8_to_bgr8(lab8: vec3<f32>) -> vec3<u32> {
    let l = lab8.x * 100.0 / 255.0;
    let a = lab8.y - 128.0;
    let bb = lab8.z - 128.0;
    var y: f32;
    var fy: f32;
    if (l <= 7.999625) {
        y = l / 903.3;
        fy = 7.787 * y + 16.0 / 116.0;
    } else {
        fy = (l + 16.0) / 116.0;
        y = fy * fy * fy;
    }
    let x = lab_f_inv(a / 500.0 + fy) * 0.950456;
    let z = lab_f_inv(fy - bb / 200.0) * 1.088754;
    let rgb = clamp(
        vec3<f32>(
            3.240479 * x - 1.53715 * y - 0.498535 * z,
            -0.969256 * x + 1.875991 * y + 0.041556 * z,
            0.055648 * x - 0.204043 * y + 1.057311 * z,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let srgb = vec3<f32>(linear_to_srgb(rgb.x), linear_to_srgb(rgb.y), linear_to_srgb(rgb.z));
    let out = clamp(round(srgb * 255.0), vec3<f32>(0.0), vec3<f32>(255.0));
    return vec3<u32>(out.zyx);
}

fn group_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.y * nwg.x + wg.x;
}

// Lab of a and b; integer sums / squared sums of Lab a, Lab b and the lower-error Lab
// (a where mask == 0, else b): 18 counters per workgroup.
@compute @workgroup_size(16, 16)
fn hist_lab(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    if (li < 18u) {
        atomicStore(&acc[li], 0u);
    }
    workgroupBarrier();
    if (id.x < p.w && id.y < p.h) {
        let i = id.y * p.w + id.x;
        let la = bgr_to_lab8(vec3<f32>(f32(byte_a(3u * i)), f32(byte_a(3u * i + 1u)), f32(byte_a(3u * i + 2u))));
        let lb = bgr_to_lab8(vec3<f32>(f32(byte_b(3u * i)), f32(byte_b(3u * i + 1u)), f32(byte_b(3u * i + 2u))));
        lab[2u * i] = pack3(la);
        lab[2u * i + 1u] = pack3(lb);
        var me = lb;
        if (byte_mask(i) == 0u) {
            me = la;
        }
        for (var k = 0u; k < 3u; k++) {
            atomicAdd(&acc[k * 2u], la[k]);
            atomicAdd(&acc[k * 2u + 1u], la[k] * la[k]);
            atomicAdd(&acc[6u + k * 2u], lb[k]);
            atomicAdd(&acc[6u + k * 2u + 1u], lb[k] * lb[k]);
            atomicAdd(&acc[12u + k * 2u], me[k]);
            atomicAdd(&acc[12u + k * 2u + 1u], me[k] * me[k]);
        }
    }
    workgroupBarrier();
    if (li < 18u) {
        part[group_index(wg, nwg) * 18u + li] = atomicLoad(&acc[li]);
    }
}

// ab = blend of the normalized Labs; partial sums of (ab - 128) and its square.
// s[0] / s[1]: mean / scale of a, s[2] / s[3]: of b; f.xy: blend weights.
@compute @workgroup_size(16, 16)
fn hist_mix(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var d = vec3<f32>(0.0);
    if (id.x < p.w && id.y < p.h) {
        let i = id.y * p.w + id.x;
        let an = (unpack3(lab[2u * i]) - p.s[0].xyz) * p.s[1].xyz + 128.0;
        let bn = (unpack3(lab[2u * i + 1u]) - p.s[2].xyz) * p.s[3].xyz + 128.0;
        let v = (an * p.f.x + bn * p.f.y - 128.0) / 0.5 + 128.0;
        ab[3u * i] = v.x;
        ab[3u * i + 1u] = v.y;
        ab[3u * i + 2u] = v.z;
        d = v - 128.0;
    }
    red_s[li] = d;
    red_q[li] = d * d;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red_s[li] += red_s[li + s];
            red_q[li] += red_q[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let base = group_index(wg, nwg) * 6u;
        for (var k = 0u; k < 3u; k++) {
            part[base + k] = bitcast<u32>(red_s[0][k]);
            part[base + 3u + k] = bitcast<u32>(red_q[0][k]);
        }
    }
}

// Final Lab = round((ab - s[0]) * s[1] + s[2]), clipped to 8 bits; packed BGR into `lab`.
@compute @workgroup_size(16, 16)
fn hist_final(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let v = vec3<f32>(ab[3u * i], ab[3u * i + 1u], ab[3u * i + 2u]);
    let l = clamp(round((v - p.s[0].xyz) * p.s[1].xyz + p.s[2].xyz), vec3<f32>(0.0), vec3<f32>(255.0));
    lab[i] = pack3(lab8_to_bgr8(l));
}
