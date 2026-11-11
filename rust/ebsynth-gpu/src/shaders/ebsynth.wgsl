// EbSynth pyramid-level synthesis (WGSL port of ezsynth/engines/backends/taichi_kernels.py).
//
// Images are uint8 HWC, packed 4 channels per u32 word and zero padded to whole words.
//   src            per source pixel: WS style words, then WG guide words
//   img[0, Nt*WT)  per target pixel: WS words of the current style estimate, then WG guide words
//   img[out_off..) per target pixel: WS words of the vote output
//   img[mod_off..) per target pixel: WG words of the guide modulation map (MODULATE only)
//   fb[0, Nt)      per target pixel patch error; summed-area tables follow (NCC only)
// Padded channels carry zero data and zero weights, so they never change a sum.

override WS: u32 = 1u;          // style words per pixel
override WG: u32 = 1u;          // guide words per pixel
override R: i32 = 3;            // patch radius (patch_size / 2)
override COST: u32 = 0u;        // 0 = SSD, 1 = NCC, 2 = SAD
override BILATERAL: bool = false;
override NSTEP: i32 = 1;        // patch sampling step (n_size_step)
override MODULATE: bool = false;
override SLOTS: u32 = 1u;       // multicand alternatives per pixel (k - 1)
override WT: u32 = WS + WG;

const COST_NCC: u32 = 1u;
const COST_SAD: u32 = 2u;
const ALT_EMPTY: f32 = 1e30;
const NO_LIMIT: f32 = 1e20;
const NCC_EPS: f32 = 1e-6;

struct Params {
    sw: i32, sh: i32, tw: i32, th: i32,
    uniformity: f32, omega_best: f32, pruning: f32, stop_threshold: i32,
    nsc: u32, out_off: u32, mod_off: u32, _pad0: u32,
    s_sat: u32, s_sq: u32, t_sat: u32, t_sq: u32,
    inv_ss: f32, inv_sc: f32, _pad1: u32, _pad2: u32,
    sweights: array<vec4<f32>, 4>,
    gweights: array<vec4<f32>, 16>,
}

// Per-dispatch arguments, selected with a dynamic offset.
//   odd: propagation sweep parity / SAT image (0 source, 1 target)
//   seed: random stream id
//   arg: JFA step size or random-search radius
//   weighted: vote weighting (0 plain, 1 weighted)
struct Step { odd: u32, seed: u32, arg: i32, weighted: u32 }

struct Alt { sx: i32, sy: i32, err: f32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> img: array<u32>;
@group(0) @binding(3) var<storage, read_write> nnf: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> fb: array<f32>;
@group(0) @binding(5) var<storage, read_write> omega: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> mask: array<u32>; // [0, Nt) active, [Nt, 2Nt) scratch
@group(0) @binding(7) var<storage, read_write> alt: array<Alt>;
@group(1) @binding(0) var<uniform> S: Step;

// ---------------------------------------------------------------- helpers

fn bytes4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u), f32(w >> 24u));
}

// Truncating float -> uint8 of values already in [0, 255].
fn pack4(v: vec4<f32>) -> u32 {
    let c = vec4<u32>(v);
    return c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u);
}

fn src_px(x: i32, y: i32) -> u32 {
    return u32(y * P.sw + x) * WT;
}

fn tgt_px(x: i32, y: i32) -> u32 {
    return u32(y * P.tw + x) * WT;
}

fn word_sum(v: vec4<f32>, s: f32) -> f32 {
    return (((s + v.x) + v.y) + v.z) + v.w;
}

// Mean of the style channels (Taichi: sum over channels, then / NSC).
fn src_gray(x: i32, y: i32) -> f32 {
    let b = src_px(x, y);
    var s = 0.0;
    for (var k = 0u; k < WS; k++) {
        s = word_sum(bytes4(src[b + k]), s);
    }
    return s / f32(P.nsc);
}

fn tgt_gray(x: i32, y: i32) -> f32 {
    let b = tgt_px(x, y);
    var s = 0.0;
    for (var k = 0u; k < WS; k++) {
        s = word_sum(bytes4(img[b + k]), s);
    }
    return s / f32(P.nsc);
}

fn mod_word(x: i32, y: i32, k: u32) -> vec4<f32> {
    return bytes4(img[P.mod_off + u32(y * P.tw + x) * WG + k]) / 255.0;
}

fn inside_target(g: vec3<u32>) -> bool {
    return i32(g.x) < P.tw && i32(g.y) < P.th;
}

fn valid_center(c: vec2<i32>) -> bool {
    return c.x >= R && c.x < P.sw - R && c.y >= R && c.y < P.sh - R;
}

// ---------------------------------------------------------------- costs

// compute_patch_ssd: SSD (or SAD) of style + modulated guide channels, optional bilateral
// weights. Returns as soon as the running sum exceeds ``ebest``.
fn cost_ssd(sx: i32, sy: i32, tx: i32, ty: i32, ebest: f32) -> f32 {
    var e = 0.0;
    var s_center = 0.0;
    var t_center = 0.0;
    if (BILATERAL) {
        s_center = src_gray(sx, sy);
        t_center = tgt_gray(tx, ty);
    }
    let steps = (2 * R) / NSTEP + 1;
    for (var i = 0; i < steps; i++) {
        let px = -R + i * NSTEP;
        let isx = clamp(sx + px, 0, P.sw - 1);
        let itx = clamp(tx + px, 0, P.tw - 1);
        for (var j = 0; j < steps; j++) {
            let py = -R + j * NSTEP;
            let isy = clamp(sy + py, 0, P.sh - 1);
            let ity = clamp(ty + py, 0, P.th - 1);
            var w = 1.0;
            if (BILATERAL) {
                let ds = src_gray(isx, isy) - s_center;
                let dt = tgt_gray(itx, ity) - t_center;
                w = exp(f32(px * px + py * py) * P.inv_ss + (ds * ds + dt * dt) * P.inv_sc);
            }
            let sb = src_px(isx, isy);
            let tb = tgt_px(itx, ity);
            for (var k = 0u; k < WS; k++) {
                let d = bytes4(src[sb + k]) - bytes4(img[tb + k]);
                let sw = P.sweights[k];
                if (COST == COST_SAD) {
                    let a = abs(d);
                    e += w * sw.x * a.x;
                    e += w * sw.y * a.y;
                    e += w * sw.z * a.z;
                    e += w * sw.w * a.w;
                } else {
                    e += w * sw.x * d.x * d.x;
                    e += w * sw.y * d.y * d.y;
                    e += w * sw.z * d.z * d.z;
                    e += w * sw.w * d.w * d.w;
                }
                if (e > ebest) { return e; }
            }
            for (var k = 0u; k < WG; k++) {
                let d = bytes4(src[sb + WS + k]) - bytes4(img[tb + WS + k]);
                let gw = P.gweights[k];
                var m = vec4<f32>(1.0);
                if (MODULATE) { m = mod_word(itx, ity, k); }
                if (COST == COST_SAD) {
                    let a = abs(d);
                    e += w * gw.x * m.x * a.x;
                    e += w * gw.y * m.y * a.y;
                    e += w * gw.z * m.z * a.z;
                    e += w * gw.w * m.w * a.w;
                } else {
                    e += w * gw.x * m.x * d.x * d.x;
                    e += w * gw.y * m.y * d.y * d.y;
                    e += w * gw.z * m.z * d.z * d.z;
                    e += w * gw.w * m.w * d.w * d.w;
                }
                if (e > ebest) { return e; }
            }
        }
    }
    return e;
}

fn query_sat(off: u32, x1_: i32, y1_: i32, x2_: i32, y2_: i32, w: i32, h: i32) -> f32 {
    let x1 = max(x1_, 0);
    let y1 = max(y1_, 0);
    let x2 = min(x2_, w - 1);
    let y2 = min(y2_, h - 1);
    let br = fb[off + u32(y2 * w + x2)];
    var bl = 0.0;
    var tr = 0.0;
    var tl = 0.0;
    if (x1 > 0) { bl = fb[off + u32(y2 * w + x1 - 1)]; }
    if (y1 > 0) { tr = fb[off + u32((y1 - 1) * w + x2)]; }
    if (x1 > 0 && y1 > 0) { tl = fb[off + u32((y1 - 1) * w + x1 - 1)]; }
    return br - bl - tr + tl;
}

fn guide_term(sb: u32, tb: u32, itx: i32, ity: i32, w: f32) -> f32 {
    var g = 0.0;
    for (var k = 0u; k < WG; k++) {
        let d = bytes4(src[sb + WS + k]) - bytes4(img[tb + WS + k]);
        let gw = P.gweights[k];
        var m = vec4<f32>(1.0);
        if (MODULATE) { m = mod_word(itx, ity, k); }
        g += w * gw.x * m.x * d.x * d.x;
        g += w * gw.y * m.y * d.y * d.y;
        g += w * gw.z * m.z * d.z * d.z;
        g += w * gw.w * m.w * d.w * d.w;
    }
    return g;
}

fn pixel_gray(base: u32, from_src: bool) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < WS; k++) {
        if (from_src) {
            s = word_sum(bytes4(src[base + k]), s);
        } else {
            s = word_sum(bytes4(img[base + k]), s);
        }
    }
    return s;
}

// compute_patch_ncc: NCC of the gray style patches + SSD of the (modulated) guides.
fn cost_ncc(sx: i32, sy: i32, tx: i32, ty: i32) -> f32 {
    let nsc = f32(P.nsc);
    if (!BILATERAL && NSTEP == 1) {
        let n = f32((2 * R + 1) * (2 * R + 1));
        let sum_s = query_sat(P.s_sat, sx - R, sy - R, sx + R, sy + R, P.sw, P.sh);
        let sum_sq_s = query_sat(P.s_sq, sx - R, sy - R, sx + R, sy + R, P.sw, P.sh);
        let sum_t = query_sat(P.t_sat, tx - R, ty - R, tx + R, ty + R, P.tw, P.th);
        let sum_sq_t = query_sat(P.t_sq, tx - R, ty - R, tx + R, ty + R, P.tw, P.th);
        let mean_s = sum_s / n;
        let mean_t = sum_t / n;
        let std_s = sqrt(max(0.0, sum_sq_s / n - mean_s * mean_s));
        let std_t = sqrt(max(0.0, sum_sq_t / n - mean_t * mean_t));
        var sum_st = 0.0;
        var guide = 0.0;
        for (var py = -R; py <= R; py++) {
            let isy = clamp(sy + py, 0, P.sh - 1);
            let ity = clamp(ty + py, 0, P.th - 1);
            for (var px = -R; px <= R; px++) {
                let isx = clamp(sx + px, 0, P.sw - 1);
                let itx = clamp(tx + px, 0, P.tw - 1);
                let sb = src_px(isx, isy);
                let tb = tgt_px(itx, ity);
                sum_st += (pixel_gray(sb, true) / nsc) * (pixel_gray(tb, false) / nsc);
                guide += guide_term(sb, tb, itx, ity, 1.0);
            }
        }
        let cov = sum_st / n - mean_s * mean_t;
        var ncc = 0.0;
        if (std_s > NCC_EPS && std_t > NCC_EPS) { ncc = cov / (std_s * std_t); }
        ncc = clamp(ncc, -1.0, 1.0);
        return (1.0 - ncc) * P.sweights[0].x * n + guide;
    }

    var sum_s = 0.0;
    var sum_sq_s = 0.0;
    var sum_t = 0.0;
    var sum_sq_t = 0.0;
    var sum_st = 0.0;
    var sum_w = 0.0;
    var guide = 0.0;
    var w_center = 1.0;
    if (BILATERAL) {
        // Taichi weights every sample by the centre colours (its kernel's behaviour).
        let dc = src_gray(sx, sy) - tgt_gray(tx, ty);
        w_center = dc * dc * P.inv_sc;
    }
    let steps = (2 * R) / NSTEP + 1;
    for (var i = 0; i < steps; i++) {
        let px = -R + i * NSTEP;
        let isx = clamp(sx + px, 0, P.sw - 1);
        let itx = clamp(tx + px, 0, P.tw - 1);
        for (var j = 0; j < steps; j++) {
            let py = -R + j * NSTEP;
            let isy = clamp(sy + py, 0, P.sh - 1);
            let ity = clamp(ty + py, 0, P.th - 1);
            var w = 1.0;
            if (BILATERAL) {
                w = exp(f32(px * px + py * py) * P.inv_ss + w_center);
            }
            let sb = src_px(isx, isy);
            let tb = tgt_px(itx, ity);
            let s_val = pixel_gray(sb, true) / nsc;
            let t_val = pixel_gray(tb, false) / nsc;
            sum_s += s_val * w;
            sum_sq_s += s_val * s_val * w;
            sum_t += t_val * w;
            sum_sq_t += t_val * t_val * w;
            sum_st += s_val * t_val * w;
            sum_w += w;
            guide += guide_term(sb, tb, itx, ity, w);
        }
    }
    if (sum_w > NCC_EPS) {
        let mean_s = sum_s / sum_w;
        let mean_t = sum_t / sum_w;
        let std_s = sqrt(max(0.0, sum_sq_s / sum_w - mean_s * mean_s));
        let std_t = sqrt(max(0.0, sum_sq_t / sum_w - mean_t * mean_t));
        let cov = sum_st / sum_w - mean_s * mean_t;
        var ncc = 0.0;
        if (std_s > NCC_EPS && std_t > NCC_EPS) { ncc = cov / (std_s * std_t); }
        ncc = clamp(ncc, -1.0, 1.0);
        return (1.0 - ncc) * P.sweights[0].x * sum_w + guide;
    }
    return guide;
}

fn patch_cost(sx: i32, sy: i32, tx: i32, ty: i32, ebest: f32) -> f32 {
    if (COST == COST_NCC) {
        return cost_ncc(sx, sy, tx, ty);
    }
    return cost_ssd(sx, sy, tx, ty, ebest);
}

// ---------------------------------------------------------------- uniformity

fn get_omega(c: vec2<i32>) -> f32 {
    var s = 0;
    for (var py = -R; py <= R; py++) {
        let y = clamp(c.y + py, 0, P.sh - 1);
        for (var px = -R; px <= R; px++) {
            let x = clamp(c.x + px, 0, P.sw - 1);
            s += atomicLoad(&omega[y * P.sw + x]);
        }
    }
    return f32(s);
}

fn omega_add(c: vec2<i32>, d: i32) {
    for (var py = -R; py <= R; py++) {
        let y = clamp(c.y + py, 0, P.sh - 1);
        for (var px = -R; px <= R; px++) {
            let x = clamp(c.x + px, 0, P.sw - 1);
            atomicAdd(&omega[y * P.sw + x], d);
        }
    }
}

fn omega_term(c: vec2<i32>) -> f32 {
    let pc = f32((2 * R + 1) * (2 * R + 1));
    return P.uniformity * (get_omega(c) / pc / P.omega_best);
}

// ---------------------------------------------------------------- random numbers

fn pcg(state: ptr<function, u32>) -> u32 {
    let s = *state * 747796405u + 2891336453u;
    *state = s;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rng_seed(t: u32) -> u32 {
    var s = (t * 2654435761u) ^ (S.seed * 0x9E3779B9u);
    _ = pcg(&s);
    return s;
}

fn rand_offset(state: ptr<function, u32>, r: i32) -> i32 {
    return i32(pcg(state) % u32(2 * r + 1)) - r;
}

// ---------------------------------------------------------------- multicand helpers

fn worst_alt(t: u32) -> f32 {
    var worst = alt[t * SLOTS].err;
    for (var s = 1u; s < SLOTS; s++) {
        worst = max(worst, alt[t * SLOTS + s].err);
    }
    return worst;
}

// Replace the worst alternative of pixel t with a better candidate that is distinct from
// every stored alternative and from the primary match.
fn offer_alt(t: u32, c: vec2<i32>, err: f32, primary: vec2<i32>) {
    var duplicate = all(c == primary);
    var worst_slot = 0u;
    var worst_err = alt[t * SLOTS].err;
    for (var s = 0u; s < SLOTS; s++) {
        let a = alt[t * SLOTS + s];
        if (a.sx == c.x && a.sy == c.y) { duplicate = true; }
        if (a.err > worst_err) {
            worst_err = a.err;
            worst_slot = s;
        }
    }
    if (!duplicate && err < worst_err) {
        alt[t * SLOTS + worst_slot] = Alt(c.x, c.y, err);
    }
}

// ---------------------------------------------------------------- setup kernels

@compute @workgroup_size(16, 16)
fn init_mask(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    mask[g.y * u32(P.tw) + g.x] = 255u;
}

@compute @workgroup_size(16, 16)
fn init_omega(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    omega_add(nnf[g.y * u32(P.tw) + g.x], 1);
}

@compute @workgroup_size(16, 16)
fn init_alt(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    let c = nnf[t];
    for (var s = 0u; s < SLOTS; s++) {
        alt[t * SLOTS + s] = Alt(c.x, c.y, ALT_EMPTY);
    }
}

// Summed-area tables of the gray style (S.odd: 0 = source, 1 = current target estimate).
@compute @workgroup_size(64)
fn sat_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    let on_target = S.odd != 0u;
    let w = select(P.sw, P.tw, on_target);
    let h = select(P.sh, P.th, on_target);
    let y = i32(g.x);
    if (y >= h) { return; }
    let o1 = select(P.s_sat, P.t_sat, on_target);
    let o2 = select(P.s_sq, P.t_sq, on_target);
    let nsc = f32(P.nsc);
    var s1 = 0.0;
    var s2 = 0.0;
    for (var x = 0; x < w; x++) {
        var v = 0.0;
        if (on_target) { v = pixel_gray(tgt_px(x, y), false) / nsc; } else { v = pixel_gray(src_px(x, y), true) / nsc; }
        s1 += v;
        s2 += v * v;
        fb[o1 + u32(y * w + x)] = s1;
        fb[o2 + u32(y * w + x)] = s2;
    }
}

@compute @workgroup_size(64)
fn sat_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    let on_target = S.odd != 0u;
    let w = select(P.sw, P.tw, on_target);
    let h = select(P.sh, P.th, on_target);
    let x = i32(g.x);
    if (x >= w) { return; }
    let o1 = select(P.s_sat, P.t_sat, on_target);
    let o2 = select(P.s_sq, P.t_sq, on_target);
    var s1 = 0.0;
    var s2 = 0.0;
    for (var y = 0; y < h; y++) {
        let i1 = o1 + u32(y * w + x);
        let i2 = o2 + u32(y * w + x);
        s1 += fb[i1];
        s2 += fb[i2];
        fb[i1] = s1;
        fb[i2] = s2;
    }
}

// ---------------------------------------------------------------- error map

@compute @workgroup_size(16, 16)
fn error_map(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    let c = nnf[t];
    fb[t] = patch_cost(c.x, c.y, i32(g.x), i32(g.y), NO_LIMIT);
}

// ---------------------------------------------------------------- PatchMatch

@compute @workgroup_size(16, 16)
fn patchmatch_step(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let odd = S.odd != 0u;
    let tx = select(P.tw - 1 - i32(g.x), i32(g.x), odd);
    let ty = select(P.th - 1 - i32(g.y), i32(g.y), odd);
    let t = u32(ty * P.tw + tx);
    if (mask[t] == 0u) { return; }
    let step = select(1, -1, odd);
    var best = nnf[t];
    var best_total = fb[t] + omega_term(best);
    for (var i = 0; i < 2; i++) {
        var n = vec2<i32>(tx, ty + step);
        var shift = vec2<i32>(0, step);
        if (i == 0) {
            n = vec2<i32>(tx + step, ty);
            shift = vec2<i32>(step, 0);
        }
        if (n.x >= 0 && n.x < P.tw && n.y >= 0 && n.y < P.th) {
            let c = nnf[n.y * P.tw + n.x] - shift;
            if (valid_center(c)) {
                let e = patch_cost(c.x, c.y, tx, ty, best_total);
                let total = e + omega_term(c);
                if (total < best_total) {
                    omega_add(best, -1);
                    omega_add(c, 1);
                    best = c;
                    fb[t] = e;
                    best_total = total;
                }
            }
        }
    }
    nnf[t] = best;
}

@compute @workgroup_size(16, 16)
fn random_search(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    let t = g.y * u32(P.tw) + g.x;
    if (mask[t] == 0u || (P.pruning > 0.0 && fb[t] < P.pruning)) { return; }
    var state = rng_seed(t);
    var best = nnf[t];
    var best_total = fb[t] + omega_term(best);
    var r = S.arg;
    while (r >= 1) {
        let ox = rand_offset(&state, r);
        let oy = rand_offset(&state, r);
        let c = best + vec2<i32>(ox, oy);
        if (valid_center(c)) {
            let e = patch_cost(c.x, c.y, tx, ty, best_total);
            let total = e + omega_term(c);
            if (total < best_total) {
                omega_add(best, -1);
                omega_add(c, 1);
                best = c;
                fb[t] = e;
                best_total = total;
            }
        }
        r = r / 2;
    }
    nnf[t] = best;
}

// ---------------------------------------------------------------- jump flooding

@compute @workgroup_size(16, 16)
fn jfa_step(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    let t = g.y * u32(P.tw) + g.x;
    if (mask[t] == 0u) { return; }
    let step = S.arg;
    var dirs = array<vec2<i32>, 8>(
        vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
        vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1), vec2<i32>(1, 1)
    );
    var best = nnf[t];
    var best_err = fb[t];
    var best_total = best_err + omega_term(best);
    for (var i = 0; i < 8; i++) {
        let d = dirs[i] * step;
        let n = vec2<i32>(tx, ty) + d;
        if (n.x >= 0 && n.x < P.tw && n.y >= 0 && n.y < P.th) {
            let c = nnf[n.y * P.tw + n.x] - d;
            if (valid_center(c)) {
                let e = patch_cost(c.x, c.y, tx, ty, best_total);
                let total = e + omega_term(c);
                if (total < best_total) {
                    omega_add(best, -1);
                    omega_add(c, 1);
                    best = c;
                    best_err = e;
                    best_total = total;
                }
            }
        }
    }
    nnf[t] = best;
    fb[t] = best_err;
}

// ---------------------------------------------------------------- multicand

@compute @workgroup_size(16, 16)
fn multicand_refresh(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    for (var s = 0u; s < SLOTS; s++) {
        let a = alt[t * SLOTS + s];
        if (a.err < ALT_EMPTY) {
            alt[t * SLOTS + s].err = patch_cost(a.sx, a.sy, i32(g.x), i32(g.y), NO_LIMIT);
        }
    }
}

@compute @workgroup_size(16, 16)
fn multicand_promote(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    if (mask[t] == 0u) { return; }
    var best = nnf[t];
    var best_err = fb[t];
    var best_total = best_err + omega_term(best);
    for (var s = 0u; s < SLOTS; s++) {
        let a = alt[t * SLOTS + s];
        if (a.err < ALT_EMPTY) {
            let c = vec2<i32>(a.sx, a.sy);
            let total = a.err + omega_term(c);
            if (total < best_total) {
                omega_add(best, -1);
                omega_add(c, 1);
                alt[t * SLOTS + s] = Alt(best.x, best.y, best_err);
                best = c;
                best_err = a.err;
                best_total = total;
            }
        }
    }
    nnf[t] = best;
    fb[t] = best_err;
}

// One candidate of the multicand propagation / random search: keep it as the best match
// or offer it (or the displaced best) to the alternatives.
fn multicand_try(t: u32, tx: i32, ty: i32, c: vec2<i32>, best: ptr<function, vec2<i32>>,
                 best_err: ptr<function, f32>, best_total: ptr<function, f32>) {
    let ebest = max(*best_total, worst_alt(t));
    let e = patch_cost(c.x, c.y, tx, ty, ebest);
    let total = e + omega_term(c);
    if (total < *best_total) {
        omega_add(*best, -1);
        omega_add(c, 1);
        offer_alt(t, *best, *best_err, c);
        *best = c;
        *best_err = e;
        *best_total = total;
    } else {
        offer_alt(t, c, e, *best);
    }
}

@compute @workgroup_size(16, 16)
fn multicand_step(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let odd = S.odd != 0u;
    let tx = select(P.tw - 1 - i32(g.x), i32(g.x), odd);
    let ty = select(P.th - 1 - i32(g.y), i32(g.y), odd);
    let t = u32(ty * P.tw + tx);
    if (mask[t] == 0u) { return; }
    let step = select(1, -1, odd);
    var best = nnf[t];
    var best_err = fb[t];
    var best_total = best_err + omega_term(best);
    for (var i = 0; i < 2; i++) {
        var n = vec2<i32>(tx, ty + step);
        var shift = vec2<i32>(0, step);
        if (i == 0) {
            n = vec2<i32>(tx + step, ty);
            shift = vec2<i32>(step, 0);
        }
        if (n.x >= 0 && n.x < P.tw && n.y >= 0 && n.y < P.th) {
            let nt = u32(n.y * P.tw + n.x);
            for (var s = 0u; s <= SLOTS; s++) {
                var c = nnf[nt];
                var present = true;
                if (s > 0u) {
                    let a = alt[nt * SLOTS + s - 1u];
                    c = vec2<i32>(a.sx, a.sy);
                    present = a.err < ALT_EMPTY;
                }
                c = c - shift;
                if (present && valid_center(c)) {
                    multicand_try(t, tx, ty, c, &best, &best_err, &best_total);
                }
            }
        }
    }
    nnf[t] = best;
    fb[t] = best_err;
}

@compute @workgroup_size(16, 16)
fn multicand_random_search(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    let t = g.y * u32(P.tw) + g.x;
    if (mask[t] == 0u || (P.pruning > 0.0 && fb[t] < P.pruning)) { return; }
    var state = rng_seed(t);
    var best = nnf[t];
    var best_err = fb[t];
    var best_total = best_err + omega_term(best);
    var r = S.arg;
    while (r >= 1) {
        for (var s = 0u; s <= SLOTS; s++) {
            var base = best;
            var present = true;
            if (s > 0u) {
                let a = alt[t * SLOTS + s - 1u];
                base = vec2<i32>(a.sx, a.sy);
                present = a.err < ALT_EMPTY;
            }
            let ox = rand_offset(&state, r);
            let oy = rand_offset(&state, r);
            let c = base + vec2<i32>(ox, oy);
            if (present && valid_center(c)) {
                multicand_try(t, tx, ty, c, &best, &best_err, &best_total);
            }
        }
        r = r / 2;
    }
    nnf[t] = best;
    fb[t] = best_err;
}

// ---------------------------------------------------------------- voting

fn out_index(t: u32, k: u32) -> u32 {
    return P.out_off + t * WS + k;
}

// vote_plain_kernel / vote_weighted_kernel (S.weighted selects the weights).
@compute @workgroup_size(16, 16)
fn vote(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    var acc: array<vec4<f32>, 4>;
    var wsum = 0.0;
    for (var py = -R; py <= R; py++) {
        let oy = ty - py;
        if (oy < 0 || oy >= P.th) { continue; }
        for (var px = -R; px <= R; px++) {
            let ox = tx - px;
            if (ox < 0 || ox >= P.tw) { continue; }
            let o = oy * P.tw + ox;
            let c = nnf[o];
            let sx = clamp(c.x + px, 0, P.sw - 1);
            let sy = clamp(c.y + py, 0, P.sh - 1);
            var w = 1.0;
            if (S.weighted != 0u) { w = 1.0 / (1.0 + fb[o]); }
            wsum += w;
            let sb = src_px(sx, sy);
            for (var k = 0u; k < WS; k++) {
                acc[k] += w * bytes4(src[sb + k]);
            }
        }
    }
    let t = g.y * u32(P.tw) + g.x;
    let ws = max(wsum, 1e-6);
    for (var k = 0u; k < WS; k++) {
        img[out_index(t, k)] = pack4(clamp(acc[k] / ws, vec4<f32>(0.0), vec4<f32>(255.0)));
    }
}

// voting_kernel (bilateral): weights also fall off with spatial offset and colour change.
// Like the Taichi kernel it votes only the first four style channels.
@compute @workgroup_size(16, 16)
fn vote_bilateral(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    let t = g.y * u32(P.tw) + g.x;
    let center = nnf[t];
    let s_center = src_gray(center.x, center.y);
    let t_center = tgt_gray(tx, ty);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var py = -R; py <= R; py++) {
        let oy = ty + py;
        if (oy < 0 || oy >= P.th) { continue; }
        for (var px = -R; px <= R; px++) {
            let ox = tx + px;
            if (ox < 0 || ox >= P.tw) { continue; }
            let o = oy * P.tw + ox;
            let c = nnf[o];
            let sx = c.x - px;
            let sy = c.y - py;
            if (sx >= 0 && sx < P.sw && sy >= 0 && sy < P.sh) {
                var w = 1.0;
                if (S.weighted != 0u) { w = 1.0 / (1.0 + fb[o]); }
                let ds = src_gray(sx, sy) - s_center;
                let dt = tgt_gray(tx, ty) - t_center;
                w *= exp(f32(px * px + py * py) * P.inv_ss + (ds * ds + dt * dt) * P.inv_sc);
                acc += bytes4(src[src_px(sx, sy)]) * w;
                wsum += w;
            }
        }
    }
    if (wsum > 0.0001) {
        img[out_index(t, 0u)] = pack4(clamp(acc / wsum, vec4<f32>(0.0), vec4<f32>(255.0)));
    } else {
        img[out_index(t, 0u)] = src[src_px(center.x, center.y)];
    }
}

// vote_weighted_multicand_kernel: every stored alternative is an extra weighted voter.
@compute @workgroup_size(16, 16)
fn vote_multicand(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    var acc: array<vec4<f32>, 4>;
    var wsum = 0.0;
    for (var py = -R; py <= R; py++) {
        let oy = ty - py;
        if (oy < 0 || oy >= P.th) { continue; }
        for (var px = -R; px <= R; px++) {
            let ox = tx - px;
            if (ox < 0 || ox >= P.tw) { continue; }
            let o = u32(oy * P.tw + ox);
            for (var s = 0u; s <= SLOTS; s++) {
                var c = nnf[o];
                var err = fb[o];
                if (s > 0u) {
                    let a = alt[o * SLOTS + s - 1u];
                    c = vec2<i32>(a.sx, a.sy);
                    err = a.err;
                }
                if (err < ALT_EMPTY) {
                    let sx = clamp(c.x + px, 0, P.sw - 1);
                    let sy = clamp(c.y + py, 0, P.sh - 1);
                    let w = 1.0 / (1.0 + err);
                    wsum += w;
                    let sb = src_px(sx, sy);
                    for (var k = 0u; k < WS; k++) {
                        acc[k] += w * bytes4(src[sb + k]);
                    }
                }
            }
        }
    }
    let t = g.y * u32(P.tw) + g.x;
    let ws = max(wsum, 1e-6);
    for (var k = 0u; k < WS; k++) {
        img[out_index(t, k)] = pack4(clamp(acc[k] / ws, vec4<f32>(0.0), vec4<f32>(255.0)));
    }
}

// ---------------------------------------------------------------- masks and copies

@compute @workgroup_size(16, 16)
fn eval_mask(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    var m = 0.0;
    for (var k = 0u; k < WS; k++) {
        let d = abs(bytes4(img[out_index(t, k)]) - bytes4(img[t * WT + k]));
        m = max(m, max(max(d.x, d.y), max(d.z, d.w)));
    }
    mask[u32(P.tw * P.th) + t] = select(0u, 255u, i32(m) >= P.stop_threshold);
}

@compute @workgroup_size(16, 16)
fn dilate_mask(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let tx = i32(g.x);
    let ty = i32(g.y);
    let n = P.tw * P.th;
    var v = 0u;
    for (var py = -R; py <= R && v == 0u; py++) {
        let ny = ty + py;
        if (ny < 0 || ny >= P.th) { continue; }
        for (var px = -R; px <= R; px++) {
            let nx = tx + px;
            if (nx >= 0 && nx < P.tw && mask[n + ny * P.tw + nx] > 0u) {
                v = 255u;
                break;
            }
        }
    }
    mask[ty * P.tw + tx] = v;
}

// Current estimate <- vote output.
@compute @workgroup_size(16, 16)
fn copy_prev(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    for (var k = 0u; k < WS; k++) {
        img[t * WT + k] = img[out_index(t, k)];
    }
}

// Vote output <- current estimate (levels without search/vote iterations).
@compute @workgroup_size(16, 16)
fn copy_out(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    for (var k = 0u; k < WS; k++) {
        img[out_index(t, k)] = img[t * WT + k];
    }
}

// Clear the vote output (the Taichi backend starts each level with a zero output image).
@compute @workgroup_size(16, 16)
fn clear_out(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!inside_target(g)) { return; }
    let t = g.y * u32(P.tw) + g.x;
    for (var k = 0u; k < WS; k++) {
        img[out_index(t, k)] = 0u;
    }
}
