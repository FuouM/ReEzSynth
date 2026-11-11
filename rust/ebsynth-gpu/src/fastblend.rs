//! FastBlend pyramid patch matching on the GPU (`PyramidPatchMatcherCore` semantics).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::GpuContext;
use crate::error::{Error, input_err};
use crate::gpu;

const STEP_STRIDE: u64 = 256;
const GRID: u32 = 16;
const KIND_RANDOM: u32 = 4;
const KIND_TRACK: u32 = 5;
const KIND_CURRENT: u32 = 6;
const STYLE_PLAIN: u32 = 0;
const STYLE_MEAN: u32 = 1;
const STYLE_PAIRWISE: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Initialize {
    Identity,
    Random,
}

/// `PyramidPatchMatcher` options (see `FastBlend/src/config.py`).
#[derive(Clone, Debug)]
pub struct FastBlendParams {
    pub height: u32,
    pub width: u32,
    /// Channels per image (1..=4).
    pub channels: u32,
    pub minimum_patch_size: u32,
    pub num_iter: u32,
    pub guide_weight: f32,
    pub use_mean_target_style: bool,
    pub use_pairwise_patch_error: bool,
    pub tracking_window_size: u32,
    pub initialize: Initialize,
    pub random_search_steps: u32,
    pub random_search_range: i32,
    pub seed: u64,
}

pub struct MatchOutput {
    /// `(B, H, W, 2)` (row, col) source coordinates.
    pub nnf: Vec<i32>,
    /// `(B, H, W, C)` remapped source style.
    pub target_style: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    h: i32,
    w: i32,
    c: i32,
    pad: i32,
    full_h: i32,
    full_w: i32,
    batch: i32,
    style_mode: u32,
    prev_h: i32,
    prev_w: i32,
    _p0: i32,
    _p1: i32,
    guide_weight: f32,
    _p2: f32,
    _p3: f32,
    _p4: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Step {
    r: i32,
    kind: u32,
    arg: i32,
    seed: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    ResamplePad,
    Remap,
    BatchMean,
    Unpad,
    Score,
    Candidates,
    SelectBetter,
    InitIdentity,
    InitRandom,
    UpscaleNnf,
}

/// Bind group (per level) used by an op.
#[derive(Clone, Copy)]
enum Bind {
    Resample(usize),
    RemapNnf,
    RemapCand,
    Mean,
    Unpad,
    Match,
    Init,
}

#[derive(Clone, Copy)]
enum Extent {
    Level,
    Padded,
    PaddedSingle,
}

struct Op {
    level: usize,
    kernel: Kernel,
    bind: Bind,
    step: u32,
    extent: Extent,
}

struct Pipelines {
    resample_pad: wgpu::ComputePipeline,
    remap: wgpu::ComputePipeline,
    batch_mean: wgpu::ComputePipeline,
    unpad: wgpu::ComputePipeline,
    score: wgpu::ComputePipeline,
    candidates: wgpu::ComputePipeline,
    select_better: wgpu::ComputePipeline,
    init_identity: wgpu::ComputePipeline,
    init_random: wgpu::ComputePipeline,
    upscale_nnf: wgpu::ComputePipeline,
}

/// SplitMix64: direction permutations and per-call seeds.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn permutation4(&mut self) -> [u32; 4] {
        let mut p = [0, 1, 2, 3];
        for i in (1..4).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

pub struct PyramidMatcher {
    ctx: Arc<GpuContext>,
    p: FastBlendParams,
    /// (height, width) per pyramid level, coarse to fine.
    sizes: Vec<(u32, u32)>,
    /// Patch size per iteration (largest first).
    patch_sizes: Vec<u32>,
    pad: u32,
    style_mode: u32,
    layout0: wgpu::BindGroupLayout,
    image_layout: wgpu::BindGroupLayout,
    match_layout: wgpu::BindGroupLayout,
    init_layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    rng: SplitMix,
}

fn module(ctx: &GpuContext, label: &str, body: &str) -> wgpu::ShaderModule {
    let source = format!(
        "{}\n{}",
        include_str!("shaders/fastblend_common.wgsl"),
        body
    );
    ctx.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

impl PyramidMatcher {
    /// Compile the matcher's pipelines for `p`; shader errors are reported here.
    pub async fn new(ctx: Arc<GpuContext>, p: FastBlendParams) -> Result<Self, Error> {
        if !(1..=4).contains(&p.channels) {
            return Err(input_err(format!(
                "FastBlend supports 1..=4 channels, got {}",
                p.channels
            )));
        }
        if p.num_iter == 0 || p.minimum_patch_size == 0 || p.minimum_patch_size.is_multiple_of(2) {
            return Err(input_err(
                "minimum_patch_size must be odd and num_iter >= 1",
            ));
        }
        if p.random_search_range < 0 {
            return Err(input_err("random_search_range must be >= 0"));
        }
        let max_patch = p.minimum_patch_size + (p.num_iter - 1) * 2;
        let ratio = p.height.min(p.width) as f64 / max_patch as f64;
        let levels = ratio.log2().trunc();
        if levels < 1.0 {
            return Err(input_err(format!(
                "{}x{} is too small for a {max_patch}x{max_patch} patch pyramid",
                p.width, p.height
            )));
        }
        let levels = levels as u32;
        let sizes: Vec<(u32, u32)> = (0..levels)
            .map(|l| (p.height >> (levels - 1 - l), p.width >> (levels - 1 - l)))
            .collect();
        let patch_sizes: Vec<u32> = (0..p.num_iter)
            .rev()
            .map(|i| p.minimum_patch_size + i * 2)
            .collect();
        let pad = patch_sizes[0] / 2;
        let style_mode = if p.use_mean_target_style {
            STYLE_MEAN
        } else if p.use_pairwise_patch_error {
            STYLE_PAIRWISE
        } else {
            STYLE_PLAIN
        };

        let device = &ctx.device;
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fastblend params"),
            entries: &[
                gpu::uniform_entry(0, false, std::mem::size_of::<Params>() as u64),
                gpu::uniform_entry(1, true, std::mem::size_of::<Step>() as u64),
            ],
        });
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fastblend image"),
            entries: &[
                gpu::storage_entry(0, true),
                gpu::storage_entry(1, false),
                gpu::storage_entry(2, true),
            ],
        });
        let match_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fastblend match"),
            entries: &[
                gpu::storage_entry(0, true),
                gpu::storage_entry(1, true),
                gpu::storage_entry(2, true),
                gpu::storage_entry(3, true),
                gpu::storage_entry(4, false),
                gpu::storage_entry(5, false),
                gpu::storage_entry(6, false),
            ],
        });
        let init_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fastblend init"),
            entries: &[gpu::storage_entry(0, true), gpu::storage_entry(1, false)],
        });
        let image_mod = module(
            &ctx,
            "fastblend_image",
            include_str!("shaders/fastblend_image.wgsl"),
        );
        let match_mod = module(
            &ctx,
            "fastblend_match",
            include_str!("shaders/fastblend_match.wgsl"),
        );
        let init_mod = module(
            &ctx,
            "fastblend_init",
            include_str!("shaders/fastblend_init.wgsl"),
        );
        let make = |m: &wgpu::ShaderModule, l: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(&layout0), Some(l)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            resample_pad: make(&image_mod, &image_layout, "resample_pad"),
            remap: make(&image_mod, &image_layout, "remap"),
            batch_mean: make(&image_mod, &image_layout, "batch_mean"),
            unpad: make(&image_mod, &image_layout, "unpad"),
            score: make(&match_mod, &match_layout, "score"),
            candidates: make(&match_mod, &match_layout, "candidates"),
            select_better: make(&match_mod, &match_layout, "select_better"),
            init_identity: make(&init_mod, &init_layout, "init_identity"),
            init_random: make(&init_mod, &init_layout, "init_random"),
            upscale_nnf: make(&init_mod, &init_layout, "upscale_nnf"),
        };
        gpu::checked(&ctx, || ()).await?;
        let rng = SplitMix(p.seed);
        Ok(PyramidMatcher {
            ctx,
            p,
            sizes,
            patch_sizes,
            pad,
            style_mode,
            layout0,
            image_layout,
            match_layout,
            init_layout,
            pipelines,
            rng,
        })
    }

    /// Number of pyramid levels.
    pub fn levels(&self) -> usize {
        self.sizes.len()
    }

    fn pipeline(&self, k: Kernel) -> &wgpu::ComputePipeline {
        let p = &self.pipelines;
        match k {
            Kernel::ResamplePad => &p.resample_pad,
            Kernel::Remap => &p.remap,
            Kernel::BatchMean => &p.batch_mean,
            Kernel::Unpad => &p.unpad,
            Kernel::Score => &p.score,
            Kernel::Candidates => &p.candidates,
            Kernel::SelectBetter => &p.select_better,
            Kernel::InitIdentity => &p.init_identity,
            Kernel::InitRandom => &p.init_random,
            Kernel::UpscaleNnf => &p.upscale_nnf,
        }
    }

    /// The dispatch sequence of `PyramidPatchMatcherCore.estimate_nnf`; `cuts` ends each
    /// submission (one per level iteration).
    fn plan(&mut self) -> (Vec<Op>, Vec<Step>, Vec<usize>) {
        let mut ops = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        let mut cuts = Vec::new();
        let mut step = |s: Step| {
            steps.push(s);
            (steps.len() - 1) as u32
        };
        let call_seed = self.rng.next() as u32;
        let mut seed = call_seed;
        let mean = self.style_mode == STYLE_MEAN;
        let levels = self.sizes.len();
        let base = step(Step::default());
        for level in 0..levels {
            let op = |ops: &mut Vec<Op>, kernel, bind, step, extent| {
                ops.push(Op {
                    level,
                    kernel,
                    bind,
                    step,
                    extent,
                })
            };
            if level == 0 {
                let k = match self.p.initialize {
                    Initialize::Identity => Kernel::InitIdentity,
                    Initialize::Random => Kernel::InitRandom,
                };
                seed = seed.wrapping_add(1);
                let s = step(Step {
                    seed,
                    ..Step::default()
                });
                op(&mut ops, k, Bind::Init, s, Extent::Level);
            } else {
                op(
                    &mut ops,
                    Kernel::UpscaleNnf,
                    Bind::Init,
                    base,
                    Extent::Level,
                );
            }
            for i in 0..3 {
                op(
                    &mut ops,
                    Kernel::ResamplePad,
                    Bind::Resample(i),
                    base,
                    Extent::Padded,
                );
            }
            for &ps in &self.patch_sizes {
                let r = ((ps - 1) / 2) as i32;
                op(
                    &mut ops,
                    Kernel::Remap,
                    Bind::RemapNnf,
                    step(Step {
                        r,
                        ..Step::default()
                    }),
                    Extent::Level,
                );
                let candidate = |ops: &mut Vec<Op>, s: u32, select: bool| {
                    op(ops, Kernel::Candidates, Bind::Match, s, Extent::Level);
                    if mean {
                        op(ops, Kernel::Remap, Bind::RemapCand, s, Extent::Level);
                        op(ops, Kernel::BatchMean, Bind::Mean, s, Extent::PaddedSingle);
                    }
                    op(ops, Kernel::Score, Bind::Match, s, Extent::Level);
                    if select {
                        op(ops, Kernel::SelectBetter, Bind::Match, s, Extent::Level);
                    }
                };
                candidate(
                    &mut ops,
                    step(Step {
                        r,
                        kind: KIND_CURRENT,
                        ..Step::default()
                    }),
                    false,
                );
                for d in self.rng.permutation4() {
                    candidate(
                        &mut ops,
                        step(Step {
                            r,
                            kind: d,
                            ..Step::default()
                        }),
                        true,
                    );
                }
                for _ in 0..self.p.random_search_steps {
                    seed = seed.wrapping_add(1);
                    let s = step(Step {
                        r,
                        kind: KIND_RANDOM,
                        arg: self.p.random_search_range,
                        seed,
                    });
                    candidate(&mut ops, s, true);
                }
                for d in 1..=self.p.tracking_window_size as i32 {
                    candidate(
                        &mut ops,
                        step(Step {
                            r,
                            kind: KIND_TRACK,
                            arg: d,
                            seed: 0,
                        }),
                        true,
                    );
                    candidate(
                        &mut ops,
                        step(Step {
                            r,
                            kind: KIND_TRACK,
                            arg: -d,
                            seed: 0,
                        }),
                        true,
                    );
                }
                cuts.push(ops.len());
            }
            if level + 1 == levels {
                let r = ((self.patch_sizes[self.patch_sizes.len() - 1] - 1) / 2) as i32;
                let s = step(Step {
                    r,
                    ..Step::default()
                });
                op(&mut ops, Kernel::Remap, Bind::RemapNnf, s, Extent::Level);
                op(&mut ops, Kernel::Unpad, Bind::Unpad, s, Extent::Level);
                cuts.push(ops.len());
            }
        }
        (ops, steps, cuts)
    }

    /// Match `batch` frames: inputs are `(B, H, W, C)` float32.
    pub async fn estimate_nnf(
        &mut self,
        batch: u32,
        source_guide: &[f32],
        target_guide: &[f32],
        source_style: &[f32],
    ) -> Result<MatchOutput, Error> {
        let (h, w, c) = (self.p.height, self.p.width, self.p.channels);
        let full = (batch * h * w * c) as usize;
        if batch == 0 {
            return Err(input_err("batch must be >= 1"));
        }
        for (name, len) in [
            ("source_guide", source_guide.len()),
            ("target_guide", target_guide.len()),
            ("source_style", source_style.len()),
        ] {
            if len != full {
                return Err(input_err(format!(
                    "{name} has {len} values, expected {batch}x{h}x{w}x{c}"
                )));
            }
        }
        if self.style_mode == STYLE_PAIRWISE && !batch.is_multiple_of(2) {
            return Err(input_err("pairwise patch error needs an even batch"));
        }
        let pad = self.pad;
        if let Some(&(lh, lw)) = self.sizes.first()
            && (lh <= pad || lw <= pad)
        {
            return Err(input_err(
                "image too small for the reflect padding of the coarsest level",
            ));
        }
        let sizes = self.sizes.clone();
        let padded_bytes = |l: usize| {
            let (lh, lw) = sizes[l];
            (batch * (lh + 2 * pad) * (lw + 2 * pad) * c) as u64 * 4
        };
        let finest = self.sizes.len() - 1;
        gpu::check_binding_size(&self.ctx, "FastBlend frames", full as u64 * 4)?;
        gpu::check_binding_size(&self.ctx, "FastBlend padded frames", padded_bytes(finest))?;

        let (ops, steps, cuts) = self.plan();
        let ctx = self.ctx.clone();
        let this = &*self;
        let (out_buf, nnf_buf) = gpu::checked(&ctx, || {
            let sg_full = gpu::storage_init(&ctx, "sg", bytemuck::cast_slice(source_guide));
            let tg_full = gpu::storage_init(&ctx, "tg", bytemuck::cast_slice(target_guide));
            let ss_full = gpu::storage_init(&ctx, "ss", bytemuck::cast_slice(source_style));
            let out_buf = gpu::storage_zeroed(&ctx, "target_style", full as u64 * 4);
            let mut step_bytes = vec![0u8; steps.len() * STEP_STRIDE as usize];
            for (i, s) in steps.iter().enumerate() {
                step_bytes[i * STEP_STRIDE as usize..i * STEP_STRIDE as usize + 16]
                    .copy_from_slice(bytemuck::bytes_of(s));
            }
            let step_buf = gpu::uniform_init(&ctx, "fastblend steps", &step_bytes);

            struct Level {
                g0: wgpu::BindGroup,
                resample: [wgpu::BindGroup; 3],
                remap_nnf: wgpu::BindGroup,
                remap_cand: Option<wgpu::BindGroup>,
                mean: Option<wgpu::BindGroup>,
                unpad: wgpu::BindGroup,
                matcher: wgpu::BindGroup,
                init: wgpu::BindGroup,
                size: (u32, u32),
            }
            let image_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, n: &wgpu::Buffer| {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fastblend image"),
                    layout: &this.image_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: n.as_entire_binding(),
                        },
                    ],
                })
            };
            let mut levels: Vec<Level> = Vec::new();
            let mut prev_nnf: Option<wgpu::Buffer> = None;
            let mut final_nnf = None;
            for (l, &(lh, lw)) in this.sizes.iter().enumerate() {
                let params = Params {
                    h: lh as i32,
                    w: lw as i32,
                    c: c as i32,
                    pad: pad as i32,
                    full_h: h as i32,
                    full_w: w as i32,
                    batch: batch as i32,
                    style_mode: this.style_mode,
                    prev_h: if l > 0 { this.sizes[l - 1].0 as i32 } else { 0 },
                    prev_w: if l > 0 { this.sizes[l - 1].1 as i32 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                    guide_weight: this.p.guide_weight,
                    _p2: 0.0,
                    _p3: 0.0,
                    _p4: 0.0,
                };
                let pbuf = gpu::uniform_init(&ctx, "fastblend params", bytemuck::bytes_of(&params));
                let g0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fastblend params"),
                    layout: &this.layout0,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pbuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &step_buf,
                                offset: 0,
                                size: wgpu::BufferSize::new(std::mem::size_of::<Step>() as u64),
                            }),
                        },
                    ],
                });
                let pb = padded_bytes(l);
                let px = (batch * lh * lw) as u64;
                let sg = gpu::storage_zeroed(&ctx, "sg_pad", pb);
                let tg = gpu::storage_zeroed(&ctx, "tg_pad", pb);
                let ss = gpu::storage_zeroed(&ctx, "ss_pad", pb);
                let ts = gpu::storage_zeroed(&ctx, "ts_pad", pb);
                let nnf = gpu::storage_zeroed(&ctx, "nnf", px * 8);
                let cand = gpu::storage_zeroed(&ctx, "cand", px * 8);
                let errs = gpu::storage_zeroed(&ctx, "errs", px * 8);
                let (remap_cand, mean, style_ref) = if this.style_mode == STYLE_MEAN {
                    let tmp = gpu::storage_zeroed(&ctx, "tmp_pad", pb);
                    let mean = gpu::storage_zeroed(&ctx, "mean_pad", pb / batch as u64);
                    (
                        Some(image_bg(&ss, &tmp, &cand)),
                        Some(image_bg(&tmp, &mean, &cand)),
                        mean,
                    )
                } else {
                    (None, None, ts.clone())
                };
                let matcher = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fastblend match"),
                    layout: &this.match_layout,
                    entries: &[&sg, &tg, &ss, &style_ref, &nnf, &cand, &errs]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| wgpu::BindGroupEntry {
                            binding: i as u32,
                            resource: b.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                });
                // The first level has no previous NNF; bind a placeholder (a buffer cannot be
                // read-only and read-write in one dispatch).
                let placeholder;
                let prev = match prev_nnf.as_ref() {
                    Some(p) => p,
                    None => {
                        placeholder = gpu::storage_zeroed(&ctx, "no previous nnf", 8);
                        &placeholder
                    }
                };
                let init = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fastblend init"),
                    layout: &this.init_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: prev.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: nnf.as_entire_binding(),
                        },
                    ],
                });
                levels.push(Level {
                    g0,
                    resample: [
                        image_bg(&sg_full, &sg, &nnf),
                        image_bg(&tg_full, &tg, &nnf),
                        image_bg(&ss_full, &ss, &nnf),
                    ],
                    remap_nnf: image_bg(&ss, &ts, &nnf),
                    remap_cand,
                    mean,
                    unpad: image_bg(&ts, &out_buf, &nnf),
                    matcher,
                    init,
                    size: (lh, lw),
                });
                final_nnf = Some(nnf.clone());
                prev_nnf = Some(nnf);
            }

            let mut start = 0;
            for &end in &cuts {
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("fastblend"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("fastblend"),
                        timestamp_writes: None,
                    });
                    for op in &ops[start..end] {
                        let lv = &levels[op.level];
                        let bg = match op.bind {
                            Bind::Resample(i) => &lv.resample[i],
                            Bind::RemapNnf => &lv.remap_nnf,
                            Bind::RemapCand => lv.remap_cand.as_ref().expect("mean mode"),
                            Bind::Mean => lv.mean.as_ref().expect("mean mode"),
                            Bind::Unpad => &lv.unpad,
                            Bind::Match => &lv.matcher,
                            Bind::Init => &lv.init,
                        };
                        pass.set_pipeline(this.pipeline(op.kernel));
                        pass.set_bind_group(0, &lv.g0, &[(op.step as u64 * STEP_STRIDE) as u32]);
                        pass.set_bind_group(1, bg, &[]);
                        let (lh, lw) = lv.size;
                        let (gw, gh, gz) = match op.extent {
                            Extent::Level => (lw, lh, batch),
                            Extent::Padded => (lw + 2 * pad, lh + 2 * pad, batch),
                            Extent::PaddedSingle => (lw + 2 * pad, lh + 2 * pad, 1),
                        };
                        pass.dispatch_workgroups(gw.div_ceil(GRID), gh.div_ceil(GRID), gz);
                    }
                }
                ctx.queue.submit([enc.finish()]);
                start = end;
            }
            (out_buf, final_nnf.expect("at least one level"))
        })
        .await?;

        let data = gpu::read_ranges(
            &ctx,
            &[
                (&out_buf, 0, full as u64 * 4),
                (&nnf_buf, 0, (batch * h * w) as u64 * 8),
            ],
        )
        .await?;
        if let Some(msg) = ctx.take_failure() {
            return Err(Error::Gpu(msg));
        }
        Ok(MatchOutput {
            target_style: bytemuck::cast_slice(&data[0]).to_vec(),
            nnf: bytemuck::cast_slice(&data[1]).to_vec(),
        })
    }
}
