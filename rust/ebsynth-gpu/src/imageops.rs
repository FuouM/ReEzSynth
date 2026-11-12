//! Image operations of the video pipeline on the GPU:
//! - backward flow warping with OpenCV 5 `cv2.remap` semantics: bilinear at exact float
//!   coordinates for 1, 3 and 4 channels and at 1/32-pixel fixed point otherwise, or
//!   nearest; reflect or constant border;
//! - forward soft splatting with pull-push hole filling (the Taichi kernels of
//!   `ezsynth/engines/backends/taichi_ops.py`);
//! - histogram blending (`hist_blender`, OpenCV path; the Lab conversion is within two
//!   8-bit levels of OpenCV's);
//! - flow-consistency occlusion masks (`compute_flow_occlusion_masks`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::GpuContext;
use crate::error::{Error, input_err};
use crate::gpu;

const WG: u32 = 16;
const HIST_COUNTERS: usize = 18;
const HIST_MIX_PARTIALS: usize = 6;

/// Border handling of [`ImageOps::warp`] (OpenCV border modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// Samples outside the image are 0 (`BORDER_CONSTANT`).
    Constant,
    /// `fedcba|abcdefgh|hgfedcb` (`BORDER_REFLECT`).
    Reflect,
}

/// Interpolation of [`ImageOps::warp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Nearest,
}

/// [`ImageOps::forward_splat`] result: image `(h, w, c)` and the splatted weight `(h, w)`
/// before hole filling.
pub struct Splat {
    pub image: Vec<f32>,
    pub weight: Vec<f32>,
}

/// Options of `compute_flow_occlusion_masks`.
#[derive(Clone, Copy, Debug)]
pub struct OcclusionParams {
    pub alpha: f32,
    pub beta: f32,
    /// Radius of the elliptical dilation; 0 disables it.
    pub dilate: u32,
    /// Also mark target pixels no rounded forward splat reaches.
    pub coverage: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    h: u32,
    w: u32,
    c: u32,
    n: u32,
    h2: u32,
    w2: u32,
    mode: u32,
    flag: u32,
    f: [f32; 4],
    s: [[f32; 4]; 4],
}

struct Pipelines {
    warp: wgpu::ComputePipeline,
    pull: wgpu::ComputePipeline,
    push: wgpu::ComputePipeline,
    normalize: wgpu::ComputePipeline,
    splat: wgpu::ComputePipeline,
    hist_lab: wgpu::ComputePipeline,
    hist_mix: wgpu::ComputePipeline,
    hist_final: wgpu::ComputePipeline,
    consistency: wgpu::ComputePipeline,
    dilate_mask: wgpu::ComputePipeline,
    scatter_cover: wgpu::ComputePipeline,
    dilate_cover: wgpu::ComputePipeline,
    combine: wgpu::ComputePipeline,
}

/// Compiled image-op pipelines on a shared device. Every call uploads its inputs, runs
/// and reads back; calls from several threads are independent.
pub struct ImageOps {
    ctx: Arc<GpuContext>,
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    /// Distinct placeholders for unused bindings (a writable buffer may be bound once).
    unused: [wgpu::Buffer; 6],
}

type Bindings<'a> = [Option<&'a wgpu::Buffer>; 6];

/// Pull-push pyramid: level sizes (finest first) and buffers of the coarser levels.
struct PullPush {
    c: u32,
    sizes: Vec<(u32, u32)>,
    coarse: Vec<(wgpu::Buffer, wgpu::Buffer)>,
}

fn groups(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(WG), h.div_ceil(WG))
}

fn check_size(h: u32, w: u32) -> Result<(), Error> {
    if h == 0 || w == 0 {
        return Err(input_err(format!(
            "image size must be positive, got {w}x{h}"
        )));
    }
    Ok(())
}

fn check_len(name: &str, len: usize, h: u32, w: u32, c: u32) -> Result<(), Error> {
    let want = h as u64 * w as u64 * c as u64;
    if len as u64 != want {
        return Err(input_err(format!(
            "{name} has {len} values, expected {h}x{w}x{c}"
        )));
    }
    Ok(())
}

fn f32_bytes(n: u64) -> u64 {
    n * 4
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytemuck::pod_collect_to_vec(bytes)
}

fn to_u32(bytes: &[u8]) -> Vec<u32> {
    bytemuck::pod_collect_to_vec(bytes)
}

/// u8 data padded to whole u32 words.
fn byte_words(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    v.resize(data.len().next_multiple_of(4).max(4), 0);
    v
}

/// (dx, dy) pairs of `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2r+1, 2r+1))`.
fn ellipse_offsets(radius: u32) -> Vec<i32> {
    if radius == 0 {
        return vec![0, 0];
    }
    let k = 2 * radius as i32 + 1;
    let r = k / 2;
    let inv_r2 = 1.0 / (r as f64 * r as f64);
    let mut out = Vec::new();
    for i in 0..k {
        let dy = i - r;
        let dx = (r as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even() as i32;
        for j in (r - dx).max(0)..(r + dx + 1).min(k) {
            out.extend([j - r, dy]);
        }
    }
    out
}

/// Mean and standard deviation of `n` values from their sum and sum of squares.
fn mean_std(sum: f64, sum_sq: f64, n: f64) -> (f64, f64) {
    let mean = sum / n;
    (mean, (sum_sq / n - mean * mean).max(0.0).sqrt())
}

fn module(ctx: &GpuContext, label: &str, body: &str) -> wgpu::ShaderModule {
    let source = format!("{}\n{}", include_str!("shaders/imageops_common.wgsl"), body);
    ctx.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

impl ImageOps {
    /// Compile the pipelines; shader errors are reported here.
    pub async fn new(ctx: Arc<GpuContext>) -> Result<Self, Error> {
        let device = &ctx.device;
        let mut entries = vec![gpu::uniform_entry(
            0,
            false,
            std::mem::size_of::<Params>() as u64,
        )];
        entries.extend((1..=6).map(|b| gpu::storage_entry(b, false)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imageops"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imageops"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = gpu::checked(&ctx, || {
            let warp_mod = module(
                &ctx,
                "imageops_warp",
                include_str!("shaders/imageops_warp.wgsl"),
            );
            let splat_mod = module(
                &ctx,
                "imageops_splat",
                include_str!("shaders/imageops_splat.wgsl"),
            );
            let hist_mod = module(
                &ctx,
                "imageops_hist",
                include_str!("shaders/imageops_hist.wgsl"),
            );
            let occ_mod = module(
                &ctx,
                "imageops_occlusion",
                include_str!("shaders/imageops_occlusion.wgsl"),
            );
            let make = |m: &wgpu::ShaderModule, entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: m,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            Pipelines {
                warp: make(&warp_mod, "warp"),
                pull: make(&warp_mod, "pull"),
                push: make(&warp_mod, "push"),
                normalize: make(&warp_mod, "normalize"),
                splat: make(&splat_mod, "splat"),
                hist_lab: make(&hist_mod, "hist_lab"),
                hist_mix: make(&hist_mod, "hist_mix"),
                hist_final: make(&hist_mod, "hist_final"),
                consistency: make(&occ_mod, "consistency"),
                dilate_mask: make(&occ_mod, "dilate_mask"),
                scatter_cover: make(&occ_mod, "scatter_cover"),
                dilate_cover: make(&occ_mod, "dilate_cover"),
                combine: make(&occ_mod, "combine"),
            }
        })
        .await?;
        let unused = std::array::from_fn(|_| gpu::storage_zeroed(&ctx, "imageops unused", 16));
        Ok(ImageOps {
            ctx,
            layout,
            pipelines,
            unused,
        })
    }

    fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        params: &Params,
        bindings: Bindings<'_>,
        (gx, gy): (u32, u32),
    ) {
        let uniform = gpu::uniform_init(&self.ctx, "imageops params", bytemuck::bytes_of(params));
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (k, b) in bindings.into_iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: k as u32 + 1,
                resource: b.unwrap_or(&self.unused[k]).as_entire_binding(),
            });
        }
        let group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("imageops"),
                layout: &self.layout,
                entries: &entries,
            });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("imageops"),
            })
    }

    fn storage(&self, label: &str, data: &[u8]) -> Result<wgpu::Buffer, Error> {
        gpu::check_binding_size(&self.ctx, label, data.len() as u64)?;
        Ok(gpu::storage_init(&self.ctx, label, data))
    }

    fn zeroed(&self, label: &str, bytes: u64) -> Result<wgpu::Buffer, Error> {
        gpu::check_binding_size(&self.ctx, label, bytes)?;
        Ok(gpu::storage_zeroed(&self.ctx, label, bytes))
    }

    /// Record into one encoder and submit it inside the error scopes.
    async fn submit(&self, record: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<(), Error> {
        gpu::checked(&self.ctx, || {
            let mut enc = self.encoder();
            record(&mut enc);
            self.ctx.queue.submit([enc.finish()]);
        })
        .await
    }

    /// `cv2.remap(src, grid + flow)`: `src` is `(h, w, c)` float32, `flow` `(h, w, 2)`.
    #[allow(clippy::too_many_arguments)]
    pub async fn warp(
        &self,
        src: &[f32],
        h: u32,
        w: u32,
        c: u32,
        flow: &[f32],
        interpolation: Interpolation,
        border: Border,
    ) -> Result<Vec<f32>, Error> {
        check_size(h, w)?;
        check_len("src", src.len(), h, w, c)?;
        check_len("flow", flow.len(), h, w, 2)?;
        let bytes = f32_bytes(src.len() as u64);
        let src_b = self.storage("warp src", bytemuck::cast_slice(src))?;
        let flow_b = self.storage("warp flow", bytemuck::cast_slice(flow))?;
        let dst = self.zeroed("warp dst", bytes)?;
        let params = Params {
            h,
            w,
            c,
            mode: (border == Border::Reflect) as u32,
            flag: (interpolation == Interpolation::Nearest) as u32,
            // OpenCV's float-coordinate SIMD kernels cover 1, 3 and 4 channels only.
            n: !matches!(c, 1 | 3 | 4) as u32,
            ..Params::default()
        };
        self.submit(|enc| {
            self.dispatch(
                enc,
                &self.pipelines.warp,
                &params,
                [Some(&src_b), Some(&flow_b), Some(&dst), None, None, None],
                groups(w, h),
            )
        })
        .await?;
        let out = gpu::read_ranges(&self.ctx, &[(&dst, 0, bytes)]).await?;
        Ok(to_f32(&out[0]))
    }

    /// Coarser pull-push levels of an `(h, w, c)` image: sizes (finest first, including
    /// the image itself) and zeroed `(color, weight)` buffers for every coarser level.
    fn pull_push_levels(&self, (h, w, c): (u32, u32, u32), levels: u32) -> Result<PullPush, Error> {
        let mut sizes = vec![(h, w)];
        for _ in 1..levels {
            let (lh, lw) = sizes[sizes.len() - 1];
            if lh <= 2 || lw <= 2 {
                break;
            }
            sizes.push((lh / 2, lw / 2));
        }
        let mut coarse = Vec::new();
        for &(lh, lw) in &sizes[1..] {
            let n = lh as u64 * lw as u64;
            coarse.push((
                self.zeroed("pull color", f32_bytes(n * c as u64))?,
                self.zeroed("pull weight", f32_bytes(n))?,
            ));
        }
        Ok(PullPush { c, sizes, coarse })
    }

    /// Pull-push passes over the finest `color` / `weight` buffers, in place.
    fn encode_pull_push(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pp: &PullPush,
        color: &wgpu::Buffer,
        weight: &wgpu::Buffer,
    ) {
        let level = |l: usize| -> (&wgpu::Buffer, &wgpu::Buffer) {
            if l == 0 {
                (color, weight)
            } else {
                (&pp.coarse[l - 1].0, &pp.coarse[l - 1].1)
            }
        };
        let sizes = &pp.sizes;
        for l in 1..sizes.len() {
            let ((sh, sw), (dh, dw)) = (sizes[l - 1], sizes[l]);
            let (src_c, src_w) = level(l - 1);
            let (dst_c, dst_w) = level(l);
            let params = Params {
                h: sh,
                w: sw,
                c: pp.c,
                h2: dh,
                w2: dw,
                ..Params::default()
            };
            self.dispatch(
                enc,
                &self.pipelines.pull,
                &params,
                [
                    Some(src_c),
                    Some(src_w),
                    Some(dst_c),
                    Some(dst_w),
                    None,
                    None,
                ],
                groups(dw, dh),
            );
        }
        for l in (0..sizes.len() - 1).rev() {
            let ((fh, fw), (ch, cw)) = (sizes[l], sizes[l + 1]);
            let (src_c, src_w) = level(l + 1);
            let (dst_c, dst_w) = level(l);
            let params = Params {
                h: ch,
                w: cw,
                c: pp.c,
                h2: fh,
                w2: fw,
                ..Params::default()
            };
            self.dispatch(
                enc,
                &self.pipelines.push,
                &params,
                [
                    Some(src_c),
                    Some(src_w),
                    Some(dst_c),
                    Some(dst_w),
                    None,
                    None,
                ],
                groups(fw, fh),
            );
        }
    }

    /// Forward soft splatting of `src` `(h, w, c)` along `flow` `(h, w, 2)`, optionally
    /// with bilateral weights from `(source, target)` guides `(h, w, 3)` and pull-push
    /// hole filling (5 levels).
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_splat(
        &self,
        src: &[f32],
        h: u32,
        w: u32,
        c: u32,
        flow: &[f32],
        guides: Option<(&[f32], &[f32])>,
        fill_holes: bool,
    ) -> Result<Splat, Error> {
        check_size(h, w)?;
        check_len("src", src.len(), h, w, c)?;
        check_len("flow", flow.len(), h, w, 2)?;
        if let Some((sg, tg)) = guides {
            check_len("src_guide", sg.len(), h, w, 3)?;
            check_len("tgt_guide", tg.len(), h, w, 3)?;
        }
        let n = h as u64 * w as u64;
        let color_bytes = f32_bytes(n * c as u64);
        let src_b = self.storage("splat src", bytemuck::cast_slice(src))?;
        let flow_b = self.storage("splat flow", bytemuck::cast_slice(flow))?;
        let guide_b = match guides {
            Some((sg, tg)) => Some((
                self.storage("splat src guide", bytemuck::cast_slice(sg))?,
                self.storage("splat tgt guide", bytemuck::cast_slice(tg))?,
            )),
            None => None,
        };
        let color = self.zeroed("splat color", color_bytes)?;
        let weight = self.zeroed("splat weight", f32_bytes(n))?;
        let raw_weight = self.zeroed("splat raw weight", f32_bytes(n))?;
        let out = self.zeroed("splat out", color_bytes)?;
        let params = Params {
            h,
            w,
            c,
            flag: guide_b.is_some() as u32,
            ..Params::default()
        };
        let pp = if fill_holes {
            Some(self.pull_push_levels((h, w, c), 5)?)
        } else {
            None
        };
        self.submit(|enc| {
            let (sg, tg) = match &guide_b {
                Some((sg, tg)) => (Some(sg), Some(tg)),
                None => (None, None),
            };
            self.dispatch(
                enc,
                &self.pipelines.splat,
                &params,
                [
                    Some(&src_b),
                    Some(&flow_b),
                    Some(&color),
                    Some(&weight),
                    sg,
                    tg,
                ],
                groups(w, h),
            );
            enc.copy_buffer_to_buffer(&weight, 0, &raw_weight, 0, f32_bytes(n));
            if let Some(pp) = &pp {
                self.encode_pull_push(enc, pp, &color, &weight);
            }
            self.dispatch(
                enc,
                &self.pipelines.normalize,
                &params,
                [Some(&color), Some(&weight), Some(&out), None, None, None],
                groups(w, h),
            );
        })
        .await?;
        let out = gpu::read_ranges(
            &self.ctx,
            &[(&out, 0, color_bytes), (&raw_weight, 0, f32_bytes(n))],
        )
        .await?;
        Ok(Splat {
            image: to_f32(&out[0]),
            weight: to_f32(&out[1]),
        })
    }

    /// Pull-push hole filling of accumulated `color` `(h, w, c)` and `weight` `(h, w)`
    /// (Taichi `Warp.run_pull_push`); returns the filled `(color, weight)`.
    pub async fn pull_push(
        &self,
        color: &[f32],
        weight: &[f32],
        h: u32,
        w: u32,
        c: u32,
        levels: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        check_size(h, w)?;
        check_len("color", color.len(), h, w, c)?;
        check_len("weight", weight.len(), h, w, 1)?;
        let color_b = self.storage("pull-push color", bytemuck::cast_slice(color))?;
        let weight_b = self.storage("pull-push weight", bytemuck::cast_slice(weight))?;
        let pp = self.pull_push_levels((h, w, c), levels)?;
        self.submit(|enc| self.encode_pull_push(enc, &pp, &color_b, &weight_b))
            .await?;
        let n = h as u64 * w as u64;
        let out = gpu::read_ranges(
            &self.ctx,
            &[
                (&color_b, 0, f32_bytes(n * c as u64)),
                (&weight_b, 0, f32_bytes(n)),
            ],
        )
        .await?;
        Ok((to_f32(&out[0]), to_f32(&out[1])))
    }

    /// Histogram blend of BGR frames `a`, `b` `(h, w, 3)` uint8 with the selection
    /// `mask` `(h, w)` (0 = a has the lower error); returns BGR `(h, w, 3)` uint8.
    #[allow(clippy::too_many_arguments)]
    pub async fn hist_blend(
        &self,
        a: &[u8],
        b: &[u8],
        mask: &[u8],
        h: u32,
        w: u32,
        weight1: f32,
        weight2: f32,
    ) -> Result<Vec<u8>, Error> {
        check_size(h, w)?;
        check_len("a", a.len(), h, w, 3)?;
        check_len("b", b.len(), h, w, 3)?;
        check_len("mask", mask.len(), h, w, 1)?;
        let n = h as u64 * w as u64;
        let (gx, gy) = groups(w, h);
        let n_groups = gx as u64 * gy as u64;
        let a_b = self.storage("hist a", &byte_words(a))?;
        let b_b = self.storage("hist b", &byte_words(b))?;
        let m_b = self.storage("hist mask", &byte_words(mask))?;
        let lab = self.zeroed("hist lab", n * 8)?;
        let ab = self.zeroed("hist ab", f32_bytes(n * 3))?;
        let part = self.zeroed("hist partials", n_groups * HIST_COUNTERS as u64 * 4)?;
        let bind = [
            Some(&a_b),
            Some(&b_b),
            Some(&m_b),
            Some(&lab),
            Some(&ab),
            Some(&part),
        ];
        let base = Params {
            h,
            w,
            ..Params::default()
        };

        self.submit(|enc| self.dispatch(enc, &self.pipelines.hist_lab, &base, bind, (gx, gy)))
            .await?;
        let counters = to_u32(
            &gpu::read_ranges(
                &self.ctx,
                &[(&part, 0, n_groups * HIST_COUNTERS as u64 * 4)],
            )
            .await?[0],
        );
        let mut sums = [0u64; HIST_COUNTERS];
        for chunk in counters.as_chunks::<HIST_COUNTERS>().0 {
            for (s, v) in sums.iter_mut().zip(chunk) {
                *s += *v as u64;
            }
        }
        // stats[image][channel] for a, b and the lower-error frame.
        let nf = n as f64;
        let stats: [[(f64, f64); 3]; 3] = std::array::from_fn(|img| {
            std::array::from_fn(|ch| {
                let k = img * 6 + ch * 2;
                mean_std(sums[k] as f64, sums[k + 1] as f64, nf)
            })
        });
        // NumPy: (lab - mean) * t_std / (std + 1e-6) + t_mean with t_std = float32(256 / 36).
        let t_std = (256.0f64 / 36.0) as f32 as f64;
        let row = |f: &dyn Fn(usize) -> f64| [f(0) as f32, f(1) as f32, f(2) as f32, 0.0];
        let mix = Params {
            f: [weight1, weight2, 0.0, 0.0],
            s: [
                row(&|ch| stats[0][ch].0),
                row(&|ch| t_std / (stats[0][ch].1 + 1e-6)),
                row(&|ch| stats[1][ch].0),
                row(&|ch| t_std / (stats[1][ch].1 + 1e-6)),
            ],
            ..base
        };
        self.submit(|enc| self.dispatch(enc, &self.pipelines.hist_mix, &mix, bind, (gx, gy)))
            .await?;
        let partials = to_f32(
            &gpu::read_ranges(
                &self.ctx,
                &[(&part, 0, n_groups * HIST_MIX_PARTIALS as u64 * 4)],
            )
            .await?[0],
        );
        let mut mix_sums = [0f64; HIST_MIX_PARTIALS];
        for chunk in partials.as_chunks::<HIST_MIX_PARTIALS>().0 {
            for (s, v) in mix_sums.iter_mut().zip(chunk) {
                *s += *v as f64;
            }
        }
        // Sums are of (ab - 128); NumPy's float32 mean / std of ab.
        let ab_stats: [(f32, f32); 3] = std::array::from_fn(|ch| {
            let (m, s) = mean_std(mix_sums[ch], mix_sums[3 + ch], nf);
            ((m + 128.0) as f32, s as f32)
        });
        let fin = Params {
            s: [
                row(&|ch| ab_stats[ch].0 as f64),
                row(&|ch| stats[2][ch].1 / (ab_stats[ch].1 + 1e-6f32) as f64),
                row(&|ch| stats[2][ch].0),
                [0.0; 4],
            ],
            ..base
        };
        self.submit(|enc| self.dispatch(enc, &self.pipelines.hist_final, &fin, bind, (gx, gy)))
            .await?;
        let packed = to_u32(&gpu::read_ranges(&self.ctx, &[(&lab, 0, n * 4)]).await?[0]);
        let mut out = Vec::with_capacity(n as usize * 3);
        for v in packed {
            out.extend([v as u8, (v >> 8) as u8, (v >> 16) as u8]);
        }
        Ok(out)
    }

    /// Encode one direction of `compute_flow_occlusion_masks`: consistency of `primary`
    /// then `secondary`, plus coverage holes of `secondary`. The result lands in `m2`.
    #[allow(clippy::too_many_arguments)]
    fn encode_occlusion(
        &self,
        enc: &mut wgpu::CommandEncoder,
        (h, w): (u32, u32),
        primary: &wgpu::Buffer,
        secondary: &wgpu::Buffer,
        masks: &[wgpu::Buffer; 3],
        offsets: &wgpu::Buffer,
        n_offsets: u32,
        p: OcclusionParams,
    ) {
        let params = Params {
            h,
            w,
            n: n_offsets,
            flag: p.coverage as u32,
            f: [p.alpha, p.beta, 0.0, 0.0],
            ..Params::default()
        };
        let bind = [
            Some(primary),
            Some(secondary),
            Some(&masks[0]),
            Some(&masks[1]),
            Some(&masks[2]),
            Some(offsets),
        ];
        let g = groups(w, h);
        let k = &self.pipelines;
        self.dispatch(enc, &k.consistency, &params, bind, g);
        self.dispatch(enc, &k.dilate_mask, &params, bind, g);
        if p.coverage {
            self.dispatch(enc, &k.scatter_cover, &params, bind, g);
            self.dispatch(enc, &k.dilate_cover, &params, bind, g);
        }
        self.dispatch(enc, &k.combine, &params, bind, g);
    }

    /// Unreliable-pixel masks (0 / 255) for one flow pair `(h, w, 2)`: the forward mask
    /// marks pixels of frame i + 1, the backward mask pixels of frame i.
    pub async fn occlusion_masks(
        &self,
        fwd: &[f32],
        bwd: &[f32],
        h: u32,
        w: u32,
        p: OcclusionParams,
    ) -> Result<(Vec<u8>, Vec<u8>), Error> {
        check_size(h, w)?;
        check_len("fwd_flow", fwd.len(), h, w, 2)?;
        check_len("bwd_flow", bwd.len(), h, w, 2)?;
        let n = h as u64 * w as u64;
        let fwd_b = self.storage("occlusion fwd", bytemuck::cast_slice(fwd))?;
        let bwd_b = self.storage("occlusion bwd", bytemuck::cast_slice(bwd))?;
        let offsets = ellipse_offsets(p.dilate);
        let offsets_b = self.storage("occlusion offsets", bytemuck::cast_slice(&offsets))?;
        let n_offsets = (offsets.len() / 2) as u32;
        let masks = |label: &str| -> Result<[wgpu::Buffer; 3], Error> {
            Ok([
                self.zeroed(label, n * 4)?,
                self.zeroed(label, n * 4)?,
                self.zeroed(label, n * 4)?,
            ])
        };
        let fwd_masks = masks("occlusion fwd masks")?;
        let bwd_masks = masks("occlusion bwd masks")?;
        self.submit(|enc| {
            // fwd mask: backward flow then forward flow; coverage of the forward flow.
            self.encode_occlusion(
                enc,
                (h, w),
                &bwd_b,
                &fwd_b,
                &fwd_masks,
                &offsets_b,
                n_offsets,
                p,
            );
            self.encode_occlusion(
                enc,
                (h, w),
                &fwd_b,
                &bwd_b,
                &bwd_masks,
                &offsets_b,
                n_offsets,
                p,
            );
        })
        .await?;
        let out = gpu::read_ranges(
            &self.ctx,
            &[(&fwd_masks[2], 0, n * 4), (&bwd_masks[2], 0, n * 4)],
        )
        .await?;
        let bytes = |b: &[u8]| to_u32(b).into_iter().map(|v| v as u8).collect::<Vec<u8>>();
        Ok((bytes(&out[0]), bytes(&out[1])))
    }
}

#[cfg(test)]
mod tests {
    use super::ellipse_offsets;

    fn grid(radius: u32) -> Vec<Vec<u8>> {
        let k = (2 * radius + 1) as usize;
        let r = radius as i32;
        let mut g = vec![vec![0u8; k]; k];
        for pair in ellipse_offsets(radius).chunks(2) {
            g[(pair[1] + r) as usize][(pair[0] + r) as usize] = 1;
        }
        g
    }

    #[test]
    fn ellipse_matches_opencv_structuring_elements() {
        // cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) / (5, 5).
        assert_eq!(grid(1), vec![vec![0, 1, 0], vec![1, 1, 1], vec![0, 1, 0]]);
        assert_eq!(
            grid(2),
            vec![
                vec![0, 0, 1, 0, 0],
                vec![1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1],
                vec![0, 0, 1, 0, 0],
            ]
        );
        assert_eq!(ellipse_offsets(0), vec![0, 0]);
    }
}
