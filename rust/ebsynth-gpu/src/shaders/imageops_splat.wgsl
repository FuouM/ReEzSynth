// Forward soft splatting (Taichi soft_splat_kernel): every source pixel adds Gaussian
// weights to the 3x3 target pixels around its flow destination.

@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f32>;
// f32 bit patterns, accumulated with compare-exchange loops.
@group(0) @binding(3) var<storage, read_write> color: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> weight: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> src_guide: array<f32>;
@group(0) @binding(6) var<storage, read_write> tgt_guide: array<f32>;

fn add_color(i: u32, v: f32) {
    var old = atomicLoad(&color[i]);
    loop {
        let r = atomicCompareExchangeWeak(&color[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

fn add_weight(i: u32, v: f32) {
    var old = atomicLoad(&weight[i]);
    loop {
        let r = atomicCompareExchangeWeak(&weight[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

// p.flag 1: bilateral weights from the 3-channel guides.
@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let fx = finite_or_zero(flow[2u * i]);
    let fy = finite_or_zero(flow[2u * i + 1u]);
    let tx = f32(id.x) + fx;
    let ty = f32(id.y) + fy;
    let ix = floor_index(tx);
    let iy = floor_index(ty);
    let sigma = 0.35 + 0.15 * exp(-sqrt(fx * fx + fy * fy) / 10.0);
    var sg = vec3<f32>(0.0);
    if (p.flag == 1u) {
        sg = vec3<f32>(
            finite_or_zero(src_guide[3u * i]),
            finite_or_zero(src_guide[3u * i + 1u]),
            finite_or_zero(src_guide[3u * i + 2u]),
        );
    }
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let gx = ix + dx;
            let gy = iy + dy;
            if (gx < 0 || gx >= i32(p.w) || gy < 0 || gy >= i32(p.h)) {
                continue;
            }
            let ddx = f32(gx) - tx;
            let ddy = f32(gy) - ty;
            var wgt = exp(-(ddx * ddx + ddy * ddy) / (2.0 * sigma * sigma));
            let t = u32(gy) * p.w + u32(gx);
            if (p.flag == 1u) {
                let tg = vec3<f32>(
                    finite_or_zero(tgt_guide[3u * t]),
                    finite_or_zero(tgt_guide[3u * t + 1u]),
                    finite_or_zero(tgt_guide[3u * t + 2u]),
                );
                let d = sg - tg;
                wgt *= exp(-dot(d, d) / (2.0 * 30.0 * 30.0));
            }
            if (wgt == wgt && wgt > 1e-4 && wgt < 1.0e20) {
                add_weight(t, wgt);
                for (var ch = 0u; ch < p.c; ch++) {
                    add_color(t * p.c + ch, wgt * finite_or_zero(src[i * p.c + ch]));
                }
            }
        }
    }
}
