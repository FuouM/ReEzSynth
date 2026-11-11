//! One EbSynth pyramid level on the GPU (same contract as the Python backends' `run_level`).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::GpuContext;
use crate::error::{Error, input_err};
use crate::gpu;

const STYLE_WORDS: usize = 4;
const GUIDE_WORDS: usize = 16;
/// Style channels supported per image.
pub const MAX_STYLE_CHANNELS: u32 = (STYLE_WORDS * 4) as u32;
/// Guide channels supported (all guides concatenated).
pub const MAX_GUIDE_CHANNELS: u32 = (GUIDE_WORDS * 4) as u32;

const STEP_STRIDE: u64 = 256;
const GRID: u32 = 16;
const LINE: u32 = 64;
const ALT_BYTES: u64 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CostFunction {
    Ssd,
    Ncc,
    Sad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    PatchMatch,
    /// Jump flooding at halving step sizes, then a fine random search.
    Jfa,
    /// PatchMatch keeping `k - 1` alternative matches per pixel (2 <= k <= 8).
    Multicand {
        k: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteMode {
    Plain,
    Weighted,
}

/// Parameters of one level (`EbsynthParamsConfig` plus per-channel weights).
#[derive(Clone, Debug)]
pub struct LevelParams {
    pub patch_size: u32,
    pub uniformity: f32,
    pub vote_mode: VoteMode,
    pub search_vote_iters: u32,
    pub patch_match_iters: u32,
    pub stop_threshold: i32,
    pub search_pruning_threshold: f32,
    pub cost: CostFunction,
    pub search: SearchMode,
    pub use_bilateral: bool,
    pub sigma_spatial: f32,
    pub sigma_color: f32,
    pub n_size_step: u32,
    /// One weight per style channel.
    pub style_weights: Vec<f32>,
    /// One weight per guide channel.
    pub guide_weights: Vec<f32>,
    /// Random stream id; vary it per level / frame.
    pub seed: u32,
}

/// Inputs of one level. Images are HWC uint8, row-major; `nnf` is `(th, tw, 2)` source
/// `(x, y)` patch centres.
pub struct LevelInput<'a> {
    pub source_width: u32,
    pub source_height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub style_channels: u32,
    pub guide_channels: u32,
    pub source_style: &'a [u8],
    pub source_guide: &'a [u8],
    pub target_guide: &'a [u8],
    /// Optional `(th, tw, guide_channels)` modulation of the guide differences (255 = 1.0).
    pub modulation: Option<&'a [u8]>,
    pub nnf: &'a [i32],
}

pub struct LevelOutput {
    /// `(th, tw, style_channels)` synthesized image.
    pub image: Vec<u8>,
    /// `(th, tw)` patch error of the final matches.
    pub error: Vec<f32>,
    /// `(th, tw, 2)` final matches.
    pub nnf: Vec<i32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sw: i32,
    sh: i32,
    tw: i32,
    th: i32,
    uniformity: f32,
    omega_best: f32,
    pruning: f32,
    stop_threshold: i32,
    nsc: u32,
    out_off: u32,
    mod_off: u32,
    _pad0: u32,
    s_sat: u32,
    s_sq: u32,
    t_sat: u32,
    t_sq: u32,
    inv_ss: f32,
    inv_sc: f32,
    _pad1: u32,
    _pad2: u32,
    sweights: [[f32; 4]; STYLE_WORDS],
    gweights: [[f32; 4]; GUIDE_WORDS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Step {
    odd: u32,
    seed: u32,
    arg: i32,
    weighted: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    InitMask,
    InitOmega,
    InitAlt,
    SatRows,
    SatCols,
    ErrorMap,
    PatchmatchStep,
    RandomSearch,
    JfaStep,
    MulticandRefresh,
    MulticandPromote,
    MulticandStep,
    MulticandRandomSearch,
    Vote,
    VoteBilateral,
    VoteMulticand,
    EvalMask,
    DilateMask,
    CopyPrev,
    CopyOut,
    ClearOut,
}

impl Kernel {
    fn entry(self) -> &'static str {
        match self {
            Kernel::InitMask => "init_mask",
            Kernel::InitOmega => "init_omega",
            Kernel::InitAlt => "init_alt",
            Kernel::SatRows => "sat_rows",
            Kernel::SatCols => "sat_cols",
            Kernel::ErrorMap => "error_map",
            Kernel::PatchmatchStep => "patchmatch_step",
            Kernel::RandomSearch => "random_search",
            Kernel::JfaStep => "jfa_step",
            Kernel::MulticandRefresh => "multicand_refresh",
            Kernel::MulticandPromote => "multicand_promote",
            Kernel::MulticandStep => "multicand_step",
            Kernel::MulticandRandomSearch => "multicand_random_search",
            Kernel::Vote => "vote",
            Kernel::VoteBilateral => "vote_bilateral",
            Kernel::VoteMulticand => "vote_multicand",
            Kernel::EvalMask => "eval_mask",
            Kernel::DilateMask => "dilate_mask",
            Kernel::CopyPrev => "copy_prev",
            Kernel::CopyOut => "copy_out",
            Kernel::ClearOut => "clear_out",
        }
    }
}

/// Shader specialization (pipeline-overridable constants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Variant {
    ws: u32,
    wg: u32,
    radius: u32,
    cost: CostFunction,
    bilateral: bool,
    nstep: u32,
    modulate: bool,
    slots: u32,
}

impl Variant {
    fn constants(&self) -> [(&'static str, f64); 8] {
        let cost = match self.cost {
            CostFunction::Ssd => 0.0,
            CostFunction::Ncc => 1.0,
            CostFunction::Sad => 2.0,
        };
        [
            ("WS", self.ws as f64),
            ("WG", self.wg as f64),
            ("R", self.radius as f64),
            ("COST", cost),
            ("BILATERAL", self.bilateral as u32 as f64),
            ("NSTEP", self.nstep as f64),
            ("MODULATE", self.modulate as u32 as f64),
            ("SLOTS", self.slots as f64),
        ]
    }
}

#[derive(Clone, Copy)]
enum Grid {
    /// One invocation per target pixel.
    Target,
    /// One invocation per row / column (summed-area tables).
    Lines(u32),
}

struct Op {
    kernel: Kernel,
    step: u32,
    grid: Grid,
}

/// Recorded level: dispatches, their step table and the submission boundaries.
struct Plan {
    ops: Vec<Op>,
    steps: Vec<Step>,
    /// Op index where each submission ends (keeps single submissions short for OS GPU watchdogs).
    cuts: Vec<usize>,
}

impl Plan {
    fn step(&mut self, s: Step) -> u32 {
        self.steps.push(s);
        (self.steps.len() - 1) as u32
    }
    fn op(&mut self, kernel: Kernel, step: u32, grid: Grid) {
        self.ops.push(Op { kernel, step, grid });
    }
    fn cut(&mut self) {
        self.cuts.push(self.ops.len());
    }
}

/// Halving JFA step sizes `max(h, w) / 2 .. 1` (`jfa_ops.jfa_step_sizes`).
fn jfa_step_sizes(h: u32, w: u32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut s = (h.max(w) / 2) as i32;
    while s >= 1 {
        out.push(s);
        s /= 2;
    }
    out
}

fn pack_words(dst: &mut [u32], stride: usize, off: usize, src: &[u8], channels: usize) {
    let words = channels.div_ceil(4);
    for (p, px) in src.chunks_exact(channels).enumerate() {
        let row = &mut dst[p * stride + off..p * stride + off + words];
        for (c, v) in px.iter().enumerate() {
            row[c / 4] |= (*v as u32) << (8 * (c % 4));
        }
    }
}

fn unpack_words(src: &[u32], words: usize, channels: usize) -> Vec<u8> {
    let n = src.len() / words;
    let mut out = vec![0u8; n * channels];
    for p in 0..n {
        for c in 0..channels {
            out[p * channels + c] = (src[p * words + c / 4] >> (8 * (c % 4))) as u8;
        }
    }
    out
}

/// EbSynth level synthesizer: pipelines are compiled per shader variant and cached.
pub struct Synthesizer {
    ctx: Arc<GpuContext>,
    module: wgpu::ShaderModule,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<(Variant, Kernel), wgpu::ComputePipeline>,
}

impl Synthesizer {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ebsynth.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ebsynth.wgsl").into()),
        });
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ebsynth level"),
            entries: &[
                gpu::uniform_entry(0, false, std::mem::size_of::<Params>() as u64),
                gpu::storage_entry(1, true),
                gpu::storage_entry(2, false),
                gpu::storage_entry(3, false),
                gpu::storage_entry(4, false),
                gpu::storage_entry(5, false),
                gpu::storage_entry(6, false),
                gpu::storage_entry(7, false),
            ],
        });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ebsynth step"),
            entries: &[gpu::uniform_entry(
                0,
                true,
                std::mem::size_of::<Step>() as u64,
            )],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ebsynth"),
            bind_group_layouts: &[Some(&layout0), Some(&layout1)],
            immediate_size: 0,
        });
        Synthesizer {
            ctx,
            module,
            layout0,
            layout1,
            pipeline_layout,
            pipelines: HashMap::new(),
        }
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Compile the pipelines `kernels` need for `variant`; shader errors are reported, and
    /// nothing invalid is cached.
    async fn ensure_pipelines(
        &mut self,
        variant: Variant,
        kernels: impl Iterator<Item = Kernel>,
    ) -> Result<(), Error> {
        let mut missing: Vec<Kernel> = Vec::new();
        for k in kernels {
            if !self.pipelines.contains_key(&(variant, k)) && !missing.contains(&k) {
                missing.push(k);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let ctx = self.ctx.clone();
        let (module, layout) = (&self.module, &self.pipeline_layout);
        let created = gpu::checked(&ctx, || {
            let constants = variant.constants();
            missing
                .iter()
                .map(|&k| {
                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some(k.entry()),
                                layout: Some(layout),
                                module,
                                entry_point: Some(k.entry()),
                                compilation_options: wgpu::PipelineCompilationOptions {
                                    constants: &constants,
                                    zero_initialize_workgroup_memory: false,
                                },
                                cache: None,
                            });
                    (k, pipeline)
                })
                .collect::<Vec<_>>()
        })
        .await?;
        for (k, pipeline) in created {
            self.pipelines.insert((variant, k), pipeline);
        }
        Ok(())
    }

    /// Run one pyramid level: search/vote iterations from `input.nnf`, then the final
    /// error map (`run_level` of the native, torch and Taichi backends).
    pub async fn run_level(
        &mut self,
        input: &LevelInput<'_>,
        p: &LevelParams,
    ) -> Result<LevelOutput, Error> {
        let v = validate(input, p)?;
        let (sw, sh, tw, th) = (
            input.source_width,
            input.source_height,
            input.target_width,
            input.target_height,
        );
        let (cs, cg) = (input.style_channels as usize, input.guide_channels as usize);
        let ws = cs.div_ceil(4);
        let wg = cg.div_ceil(4);
        let wt = ws + wg;
        let nt = (tw * th) as usize;
        let ns = (sw * sh) as usize;
        let ncc = p.cost == CostFunction::Ncc;
        let slots = match p.search {
            SearchMode::Multicand { k } => k - 1,
            _ => 0,
        };
        let variant = Variant {
            ws: ws as u32,
            wg: wg as u32,
            radius: p.patch_size / 2,
            cost: p.cost,
            bilateral: p.use_bilateral,
            nstep: p.n_size_step,
            modulate: input.modulation.is_some(),
            slots: slots.max(1),
        };

        // ---- host packing
        let mut src = vec![0u32; ns * wt];
        pack_words(&mut src, wt, 0, input.source_style, cs);
        pack_words(&mut src, wt, ws, input.source_guide, cg);
        let out_off = nt * wt;
        let mod_off = out_off + nt * ws;
        let img_words = mod_off
            + if input.modulation.is_some() {
                nt * wg
            } else {
                0
            };
        let mut img = vec![0u32; img_words];
        pack_words(&mut img, wt, ws, input.target_guide, cg);
        if let Some(m) = input.modulation {
            pack_words(&mut img[mod_off..], wg, 0, m, cg);
        }
        let (s_sat, s_sq, t_sat, t_sq) = (nt, nt + ns, nt + 2 * ns, 2 * nt + 2 * ns);
        let fb_len = if ncc { 3 * nt + 2 * ns } else { nt };

        let mut sweights = [[0f32; 4]; STYLE_WORDS];
        for (c, w) in p.style_weights.iter().enumerate() {
            sweights[c / 4][c % 4] = *w;
        }
        let mut gweights = [[0f32; 4]; GUIDE_WORDS];
        for (c, w) in p.guide_weights.iter().enumerate() {
            gweights[c / 4][c % 4] = *w;
        }
        let pc = (p.patch_size * p.patch_size) as f32;
        let params = Params {
            sw: sw as i32,
            sh: sh as i32,
            tw: tw as i32,
            th: th as i32,
            uniformity: p.uniformity,
            omega_best: ((tw * th) as f32 / (sw * sh) as f32 * pc).max(1e-6),
            pruning: p.search_pruning_threshold,
            stop_threshold: p.stop_threshold,
            nsc: cs as u32,
            out_off: out_off as u32,
            mod_off: mod_off as u32,
            _pad0: 0,
            s_sat: s_sat as u32,
            s_sq: s_sq as u32,
            t_sat: t_sat as u32,
            t_sq: t_sq as u32,
            inv_ss: -1.0 / (2.0 * p.sigma_spatial * p.sigma_spatial),
            inv_sc: -1.0 / (2.0 * p.sigma_color * p.sigma_color),
            _pad1: 0,
            _pad2: 0,
            sweights,
            gweights,
        };

        let ctx = self.ctx.clone();
        for (what, bytes) in [
            ("source images", (src.len() * 4) as u64),
            ("target images", (img.len() * 4) as u64),
            ("error / summed-area tables", (fb_len * 4) as u64),
            (
                "multicand alternatives",
                nt as u64 * slots as u64 * ALT_BYTES,
            ),
        ] {
            gpu::check_binding_size(&ctx, what, bytes)?;
        }

        let plan = plan_level(p, v, sw, sh, tw, th, slots);
        self.ensure_pipelines(variant, plan.ops.iter().map(|o| o.kernel))
            .await?;

        let level = gpu::checked(&ctx, || {
            let params_buf = gpu::uniform_init(&ctx, "params", bytemuck::bytes_of(&params));
            let src_buf = gpu::storage_init(&ctx, "src", bytemuck::cast_slice(&src));
            let img_buf = gpu::storage_init(&ctx, "img", bytemuck::cast_slice(&img));
            let nnf_buf = gpu::storage_init(&ctx, "nnf", bytemuck::cast_slice(input.nnf));
            let fb_buf = gpu::storage_zeroed(&ctx, "fb", fb_len as u64 * 4);
            let omega_buf = gpu::storage_zeroed(&ctx, "omega", ns as u64 * 4);
            let mask_buf = gpu::storage_zeroed(&ctx, "mask", nt as u64 * 8);
            let alt_buf = gpu::storage_zeroed(
                &ctx,
                "alt",
                (nt as u64 * slots as u64 * ALT_BYTES).max(ALT_BYTES),
            );
            let mut steps = vec![0u8; plan.steps.len().max(1) * STEP_STRIDE as usize];
            for (i, s) in plan.steps.iter().enumerate() {
                let o = i * STEP_STRIDE as usize;
                steps[o..o + 16].copy_from_slice(bytemuck::bytes_of(s));
            }
            let step_buf = gpu::uniform_init(&ctx, "steps", &steps);
            let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ebsynth level"),
                layout: &self.layout0,
                entries: &[
                    &params_buf,
                    &src_buf,
                    &img_buf,
                    &nnf_buf,
                    &fb_buf,
                    &omega_buf,
                    &mask_buf,
                    &alt_buf,
                ]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
            });
            let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ebsynth step"),
                layout: &self.layout1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &step_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Step>() as u64),
                    }),
                }],
            });

            let target_groups = (tw.div_ceil(GRID), th.div_ceil(GRID));
            let mut start = 0;
            for &end in &plan.cuts {
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("ebsynth"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ebsynth"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, &bg0, &[]);
                    for op in &plan.ops[start..end] {
                        pass.set_pipeline(&self.pipelines[&(variant, op.kernel)]);
                        pass.set_bind_group(1, &bg1, &[(op.step as u64 * STEP_STRIDE) as u32]);
                        match op.grid {
                            Grid::Target => {
                                pass.dispatch_workgroups(target_groups.0, target_groups.1, 1)
                            }
                            Grid::Lines(n) => pass.dispatch_workgroups(n.div_ceil(LINE), 1, 1),
                        }
                    }
                }
                ctx.queue.submit([enc.finish()]);
                start = end;
            }
            (img_buf, nnf_buf, fb_buf)
        })
        .await?;

        let (img_buf, nnf_buf, fb_buf) = level;
        let data = gpu::read_ranges(
            &ctx,
            &[
                (&img_buf, out_off as u64 * 4, (nt * ws) as u64 * 4),
                (&fb_buf, 0, nt as u64 * 4),
                (&nnf_buf, 0, nt as u64 * 8),
            ],
        )
        .await?;
        if let Some(msg) = ctx.take_failure() {
            return Err(Error::Gpu(msg));
        }
        Ok(LevelOutput {
            image: unpack_words(bytemuck::cast_slice(&data[0]), ws, cs),
            error: bytemuck::cast_slice(&data[1]).to_vec(),
            nnf: bytemuck::cast_slice(&data[2]).to_vec(),
        })
    }
}

#[derive(Clone, Copy)]
struct Validated {
    weighted: bool,
}

fn validate(input: &LevelInput<'_>, p: &LevelParams) -> Result<Validated, Error> {
    let (sw, sh, tw, th) = (
        input.source_width as usize,
        input.source_height as usize,
        input.target_width as usize,
        input.target_height as usize,
    );
    let (cs, cg) = (input.style_channels as usize, input.guide_channels as usize);
    if sw == 0 || sh == 0 || tw == 0 || th == 0 {
        return Err(input_err("image sizes must be non-zero"));
    }
    if cs == 0 || cs > MAX_STYLE_CHANNELS as usize {
        return Err(input_err(format!(
            "style needs 1..={MAX_STYLE_CHANNELS} channels, got {cs}"
        )));
    }
    if cg == 0 || cg > MAX_GUIDE_CHANNELS as usize {
        return Err(input_err(format!(
            "guides need 1..={MAX_GUIDE_CHANNELS} channels in total, got {cg}"
        )));
    }
    let check = |name: &str, got: usize, want: usize| {
        if got != want {
            Err(input_err(format!(
                "{name} has {got} elements, expected {want}"
            )))
        } else {
            Ok(())
        }
    };
    check("source_style", input.source_style.len(), sw * sh * cs)?;
    check("source_guide", input.source_guide.len(), sw * sh * cg)?;
    check("target_guide", input.target_guide.len(), tw * th * cg)?;
    if let Some(m) = input.modulation {
        check("modulation", m.len(), tw * th * cg)?;
    }
    check("nnf", input.nnf.len(), tw * th * 2)?;
    check("style_weights", p.style_weights.len(), cs)?;
    check("guide_weights", p.guide_weights.len(), cg)?;
    if p.patch_size < 3 || p.patch_size.is_multiple_of(2) {
        return Err(input_err(format!(
            "patch_size must be odd and >= 3, got {}",
            p.patch_size
        )));
    }
    let r = (p.patch_size / 2) as usize;
    if sw <= 2 * r || sh <= 2 * r {
        return Err(input_err(format!(
            "source {sw}x{sh} is smaller than one {0}x{0} patch",
            p.patch_size
        )));
    }
    if p.n_size_step == 0 {
        return Err(input_err("n_size_step must be >= 1"));
    }
    if let SearchMode::Multicand { k } = p.search
        && !(2..=8).contains(&k)
    {
        return Err(input_err(format!("multicand_k must be in 2..=8, got {k}")));
    }
    if p.use_bilateral && (p.sigma_spatial <= 0.0 || p.sigma_color <= 0.0) {
        return Err(input_err("bilateral sigmas must be > 0"));
    }
    Ok(Validated {
        weighted: p.vote_mode == VoteMode::Weighted,
    })
}

/// The dispatch sequence of `TaichiBackend.run_level`.
fn plan_level(
    p: &LevelParams,
    v: Validated,
    sw: u32,
    sh: u32,
    tw: u32,
    th: u32,
    slots: u32,
) -> Plan {
    let mut plan = Plan {
        ops: Vec::new(),
        steps: Vec::new(),
        cuts: Vec::new(),
    };
    let base = plan.step(Step::default());
    let odd = plan.step(Step {
        odd: 1,
        ..Step::default()
    });
    let weighted = plan.step(Step {
        weighted: v.weighted as u32,
        ..Step::default()
    });
    let ncc = p.cost == CostFunction::Ncc;
    let multicand = slots > 0;
    let full_radius = (sw.max(sh) / 2) as i32;
    let mut seed = p.seed.wrapping_mul(0x9E37_79B9);
    let mut next_seed = |plan: &mut Plan, arg: i32| {
        seed = seed.wrapping_add(1);
        plan.step(Step {
            seed,
            arg,
            ..Step::default()
        })
    };

    plan.op(Kernel::InitMask, base, Grid::Target);
    if ncc {
        plan.op(Kernel::SatRows, base, Grid::Lines(sh));
        plan.op(Kernel::SatCols, base, Grid::Lines(sw));
    }
    plan.op(Kernel::InitOmega, base, Grid::Target);
    let initial_vote = if p.use_bilateral {
        Kernel::VoteBilateral
    } else {
        Kernel::Vote
    };
    plan.op(initial_vote, base, Grid::Target);
    plan.op(Kernel::CopyPrev, base, Grid::Target);
    plan.op(Kernel::ClearOut, base, Grid::Target);
    if multicand {
        plan.op(Kernel::InitAlt, base, Grid::Target);
    }
    plan.cut();

    for iter in 0..p.search_vote_iters {
        if ncc {
            plan.op(Kernel::SatRows, odd, Grid::Lines(th));
            plan.op(Kernel::SatCols, odd, Grid::Lines(tw));
        }
        plan.op(Kernel::ErrorMap, base, Grid::Target);
        match p.search {
            SearchMode::Jfa => {
                for s in jfa_step_sizes(th, tw) {
                    let st = plan.step(Step {
                        arg: s,
                        ..Step::default()
                    });
                    plan.op(Kernel::JfaStep, st, Grid::Target);
                }
                let fine = (sh.max(sw) / 16) as i32;
                if fine >= 1 {
                    let st = next_seed(&mut plan, fine);
                    plan.op(Kernel::RandomSearch, st, Grid::Target);
                }
            }
            SearchMode::Multicand { .. } => {
                plan.op(Kernel::MulticandRefresh, base, Grid::Target);
                plan.op(Kernel::MulticandPromote, base, Grid::Target);
                for pm in 0..p.patch_match_iters {
                    plan.op(
                        Kernel::MulticandStep,
                        if pm % 2 == 1 { odd } else { base },
                        Grid::Target,
                    );
                }
                let st = next_seed(&mut plan, full_radius);
                plan.op(Kernel::MulticandRandomSearch, st, Grid::Target);
            }
            SearchMode::PatchMatch => {
                for pm in 0..p.patch_match_iters {
                    plan.op(
                        Kernel::PatchmatchStep,
                        if pm % 2 == 1 { odd } else { base },
                        Grid::Target,
                    );
                }
                let st = next_seed(&mut plan, full_radius);
                plan.op(Kernel::RandomSearch, st, Grid::Target);
            }
        }
        let vote = if multicand && v.weighted && !p.use_bilateral {
            Kernel::VoteMulticand
        } else if p.use_bilateral {
            Kernel::VoteBilateral
        } else {
            Kernel::Vote
        };
        plan.op(vote, weighted, Grid::Target);
        if iter + 1 < p.search_vote_iters {
            plan.op(Kernel::EvalMask, base, Grid::Target);
            plan.op(Kernel::DilateMask, base, Grid::Target);
        }
        plan.op(Kernel::CopyPrev, base, Grid::Target);
        plan.cut();
    }

    if p.search_vote_iters == 0 {
        plan.op(Kernel::CopyOut, base, Grid::Target);
    }
    if ncc {
        plan.op(Kernel::SatRows, odd, Grid::Lines(th));
        plan.op(Kernel::SatCols, odd, Grid::Lines(tw));
    }
    plan.op(Kernel::ErrorMap, base, Grid::Target);
    plan.cut();
    plan
}
