// Matching kernels: candidate fields, patch errors and selection.

@group(1) @binding(0) var<storage, read> source_guide: array<f32>;
@group(1) @binding(1) var<storage, read> target_guide: array<f32>;
@group(1) @binding(2) var<storage, read> source_style: array<f32>;
@group(1) @binding(3) var<storage, read> style_ref: array<f32>; // remapped style or its batch mean
@group(1) @binding(4) var<storage, read_write> nnf: array<vec2<i32>>;
@group(1) @binding(5) var<storage, read_write> cand: array<vec2<i32>>;
@group(1) @binding(6) var<storage, read_write> errs: array<f32>; // [current | candidate]

const KIND_RANDOM: u32 = 4u;
const KIND_TRACK: u32 = 5u;
const KIND_CURRENT: u32 = 6u;

fn clamp_match(m: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(clamp(m.x, 0, P.h - 1), clamp(m.y, 0, P.w - 1));
}

// patch_error_kernel(source_guide, nnf, target_guide).
fn guide_error(b: i32, y: i32, x: i32, m: vec2<i32>, r: i32) -> f32 {
    var e = 0.0;
    for (var py = -r; py <= r; py++) {
        for (var px = -r; px <= r; px++) {
            let t = padded_index(b, y + P.pad + py, x + P.pad + px);
            let s = padded_index(b, m.x + P.pad + py, m.y + P.pad + px);
            for (var ch = 0; ch < P.c; ch++) {
                let d = target_guide[t + ch] - source_guide[s + ch];
                e += d * d;
            }
        }
    }
    return e;
}

// patch_error_kernel against the remapped style (or its batch mean), or
// pairwise_patch_error_kernel between the two frames of a pair.
fn style_error(b: i32, y: i32, x: i32, m: vec2<i32>, r: i32) -> f32 {
    var e = 0.0;
    if (P.style_mode == STYLE_PAIRWISE) {
        let a = b - (b % 2);
        let ma = cand[nnf_index(a, y, x)];
        let mb = cand[nnf_index(a + 1, y, x)];
        for (var py = -r; py <= r; py++) {
            for (var px = -r; px <= r; px++) {
                let sa = padded_index(a, ma.x + P.pad + py, ma.y + P.pad + px);
                let sb = padded_index(a + 1, mb.x + P.pad + py, mb.y + P.pad + px);
                for (var ch = 0; ch < P.c; ch++) {
                    let d = source_style[sa + ch] - source_style[sb + ch];
                    e += d * d;
                }
            }
        }
        return e;
    }
    let rb = select(b, 0, P.style_mode == STYLE_MEAN);
    for (var py = -r; py <= r; py++) {
        for (var px = -r; px <= r; px++) {
            let t = padded_index(rb, y + P.pad + py, x + P.pad + px);
            let s = padded_index(b, m.x + P.pad + py, m.y + P.pad + px);
            for (var ch = 0; ch < P.c; ch++) {
                let d = style_ref[t + ch] - source_style[s + ch];
                e += d * d;
            }
        }
    }
    return e;
}

// Error of the candidate field; written to the current half for KIND_CURRENT.
@compute @workgroup_size(16, 16, 1)
fn score(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    let i = nnf_index(b, y, x);
    let m = cand[i];
    let e = guide_error(b, y, x, m, S.r) * P.guide_weight + style_error(b, y, x, m, S.r);
    let half = select(P.batch * P.h * P.w, 0, S.kind == KIND_CURRENT);
    errs[half + i] = e;
}

// Candidate field from the current NNF: neighbour shift (kinds 0-3), random jitter, temporal
// track, or a copy of the current field.
@compute @workgroup_size(16, 16, 1)
fn candidates(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    var m: vec2<i32>;
    switch S.kind {
        case 0u: { m = clamp_match(nnf[nnf_index(b, max(y - 1, 0), x)] + vec2<i32>(1, 0)); }
        case 1u: { m = clamp_match(nnf[nnf_index(b, y, max(x - 1, 0))] + vec2<i32>(0, 1)); }
        case 2u: { m = clamp_match(nnf[nnf_index(b, min(y + 1, P.h - 1), x)] - vec2<i32>(1, 0)); }
        case 3u: { m = clamp_match(nnf[nnf_index(b, y, min(x + 1, P.w - 1))] - vec2<i32>(0, 1)); }
        case 4u: {
            var st = rng(b, y, x);
            let span = u32(2 * S.arg + 1);
            let dy = i32(pcg(&st) % span) - S.arg;
            let dx = i32(pcg(&st) % span) - S.arg;
            m = clamp_match(nnf[nnf_index(b, y, x)] + vec2<i32>(dy, dx));
        }
        case 5u: {
            // The match of frame b + arg, clamped to the batch; with pairwise errors the even
            // and odd frames form separate sequences.
            var sb = clamp(b + S.arg, 0, P.batch - 1);
            if (P.style_mode == STYLE_PAIRWISE) {
                let n = (P.batch - (b % 2) + 1) / 2;
                sb = 2 * clamp(b / 2 + S.arg, 0, n - 1) + (b % 2);
            }
            m = nnf[nnf_index(sb, y, x)];
        }
        default: { m = nnf[nnf_index(b, y, x)]; }
    }
    cand[nnf_index(b, y, x)] = m;
}

// Keep candidates with a strictly lower error.
@compute @workgroup_size(16, 16, 1)
fn select_better(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let i = nnf_index(i32(g.z), i32(g.y), i32(g.x));
    let ce = errs[P.batch * P.h * P.w + i];
    if (ce < errs[i]) {
        nnf[i] = cand[i];
        errs[i] = ce;
    }
}
