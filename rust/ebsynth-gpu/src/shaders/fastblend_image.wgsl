// Image kernels: resample + pad, remap, batch mean, unpad.

@group(1) @binding(0) var<storage, read> img_in: array<f32>;
@group(1) @binding(1) var<storage, read_write> img_out: array<f32>;
@group(1) @binding(2) var<storage, read> img_nnf: array<vec2<i32>>;

fn reflect(i: i32, n: i32) -> i32 {
    if (i < 0) { return -i; }
    if (i >= n) { return 2 * (n - 1) - i; }
    return i;
}

// Area resize (adaptive average pooling, torch `mode="area"`) of the full-size input to the
// level size, written with a reflect border of `pad`.
@compute @workgroup_size(16, 16, 1)
fn resample_pad(@builtin(global_invocation_id) g: vec3<u32>) {
    let hp = P.h + 2 * P.pad;
    let wp = P.w + 2 * P.pad;
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    if (x >= wp || y >= hp || b >= P.batch) { return; }
    let ly = reflect(y - P.pad, P.h);
    let lx = reflect(x - P.pad, P.w);
    let y0 = (ly * P.full_h) / P.h;
    let y1 = ((ly + 1) * P.full_h + P.h - 1) / P.h;
    let x0 = (lx * P.full_w) / P.w;
    let x1 = ((lx + 1) * P.full_w + P.w - 1) / P.w;
    let area = f32((y1 - y0) * (x1 - x0));
    let o = padded_index(b, y, x);
    for (var ch = 0; ch < P.c; ch++) {
        var s = 0.0;
        for (var yy = y0; yy < y1; yy++) {
            for (var xx = x0; xx < x1; xx++) {
                s += img_in[((b * P.full_h + yy) * P.full_w + xx) * P.c + ch];
            }
        }
        img_out[o + ch] = s / area;
    }
}

// remap_kernel: average of the source pixels voted for by the matches of all patches that
// cover (y, x). Writes only the interior; the border keeps its (zero) content.
@compute @workgroup_size(16, 16, 1)
fn remap(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    let r = S.r;
    let min_py = select(-r, -y, y < r);
    let max_py = select(r, P.h - 1 - y, y + r > P.h - 1);
    let min_px = select(-r, -x, x < r);
    let max_px = select(r, P.w - 1 - x, x + r > P.w - 1);
    var acc = vec4<f32>(0.0);
    var num = 0.0;
    for (var py = min_py; py <= max_py; py++) {
        for (var px = min_px; px <= max_px; px++) {
            let m = img_nnf[nnf_index(b, y + py, x + px)];
            let sy = m.x - py;
            let sx = m.y - px;
            if (sy >= 0 && sy < P.h && sx >= 0 && sx < P.w) {
                num += 1.0;
                let s = padded_index(b, sy + P.pad, sx + P.pad);
                for (var ch = 0; ch < P.c; ch++) {
                    acc[ch] += img_in[s + ch];
                }
            }
        }
    }
    let o = padded_index(b, y + P.pad, x + P.pad);
    for (var ch = 0; ch < P.c; ch++) {
        img_out[o + ch] = select(acc[ch], acc[ch] / num, num > 0.0);
    }
}

// Mean over the batch of a padded image stack (one padded image out).
@compute @workgroup_size(16, 16, 1)
fn batch_mean(@builtin(global_invocation_id) g: vec3<u32>) {
    let hp = P.h + 2 * P.pad;
    let wp = P.w + 2 * P.pad;
    let x = i32(g.x);
    let y = i32(g.y);
    if (x >= wp || y >= hp || g.z != 0u) { return; }
    for (var ch = 0; ch < P.c; ch++) {
        var s = 0.0;
        for (var b = 0; b < P.batch; b++) {
            s += img_in[padded_index(b, y, x) + ch];
        }
        img_out[padded_index(0, y, x) + ch] = s / f32(P.batch);
    }
}

// Interior of a padded stack -> unpadded (B, h, w, C).
@compute @workgroup_size(16, 16, 1)
fn unpad(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!in_level(g)) { return; }
    let x = i32(g.x);
    let y = i32(g.y);
    let b = i32(g.z);
    let s = padded_index(b, y + P.pad, x + P.pad);
    let o = nnf_index(b, y, x) * P.c;
    for (var ch = 0; ch < P.c; ch++) {
        img_out[o + ch] = img_in[s + ch];
    }
}
