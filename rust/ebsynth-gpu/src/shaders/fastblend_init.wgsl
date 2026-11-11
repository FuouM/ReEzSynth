// NNF initialization of the first level and upscaling between levels.

@group(1) @binding(0) var<storage, read> prev_nnf: array<vec2<i32>>;
@group(1) @binding(1) var<storage, read_write> nnf_out: array<vec2<i32>>;

@compute @workgroup_size(16, 16, 1)
fn init_identity(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    nnf_out[nnf_index(i32(g.z), i32(g.y), i32(g.x))] = vec2<i32>(i32(g.y), i32(g.x));
}

@compute @workgroup_size(16, 16, 1)
fn init_random(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let b = i32(g.z);
    let y = i32(g.y);
    let x = i32(g.x);
    var st = rng(b, y, x);
    let ry = i32(pcg(&st) % u32(P.h));
    let rx = i32(pcg(&st) % u32(P.w));
    nnf_out[nnf_index(b, y, x)] = vec2<i32>(ry, rx);
}

// update_nnf: 2x nearest upscale of the previous level's NNF times 2 (odd columns +1 on the
// row channel, odd rows +1 on the column channel, as in PyramidPatchMatcherCore), then a
// bilinear resize to the level size when the sizes differ.
fn upscaled(b: i32, yy: i32, xx: i32) -> vec2<f32> {
    let m = prev_nnf[(b * P.prev_h + yy / 2) * P.prev_w + xx / 2];
    return vec2<f32>(f32(m.x * 2 + (xx & 1)), f32(m.y * 2 + (yy & 1)));
}

fn tap(o: i32, in_size: i32, out_size: i32) -> vec3<f32> {
    let s = max((f32(o) + 0.5) * (f32(in_size) / f32(out_size)) - 0.5, 0.0);
    let i0 = i32(s);
    let i1 = select(i0, i0 + 1, i0 < in_size - 1);
    return vec3<f32>(f32(i0), f32(i1), s - f32(i0));
}

@compute @workgroup_size(16, 16, 1)
fn upscale_nnf(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    let uh = 2 * P.prev_h;
    let uw = 2 * P.prev_w;
    if (uh == P.h && uw == P.w) {
        nnf_out[nnf_index(b, y, x)] = vec2<i32>(upscaled(b, y, x));
        return;
    }
    let ty = tap(y, uh, P.h);
    let tx = tap(x, uw, P.w);
    let y0 = i32(ty.x);
    let y1 = i32(ty.y);
    let x0 = i32(tx.x);
    let x1 = i32(tx.y);
    let v = (1.0 - ty.z) * ((1.0 - tx.z) * upscaled(b, y0, x0) + tx.z * upscaled(b, y0, x1))
        + ty.z * ((1.0 - tx.z) * upscaled(b, y1, x0) + tx.z * upscaled(b, y1, x1));
    let m = vec2<i32>(v);
    nnf_out[nnf_index(b, y, x)] = vec2<i32>(clamp(m.x, 0, P.h - 1), clamp(m.y, 0, P.w - 1));
}
