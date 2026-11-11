//! Whole-image EbSynth (the `EbsynthEngine` pyramid loop) on top of [`Synthesizer`].

use crate::ebsynth::{CostFunction, LevelInput, LevelParams, SearchMode, Synthesizer, VoteMode};
use crate::error::{Error, input_err};

/// One guide pair: source (style-aligned) and target images, both HWC uint8.
pub struct Guide<'a> {
    pub source: &'a [u8],
    pub target: &'a [u8],
    pub channels: u32,
    pub weight: f32,
}

/// `RunConfig` / `EbsynthParamsConfig` options used by image synthesis.
#[derive(Clone, Debug)]
pub struct SynthParams {
    pub pyramid_levels: u32,
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
    pub extra_pass_3x3: bool,
    pub seed: u32,
}

impl Default for SynthParams {
    fn default() -> Self {
        SynthParams {
            pyramid_levels: 6,
            patch_size: 7,
            uniformity: 3500.0,
            vote_mode: VoteMode::Weighted,
            search_vote_iters: 12,
            patch_match_iters: 6,
            stop_threshold: 5,
            search_pruning_threshold: 50.0,
            cost: CostFunction::Ssd,
            search: SearchMode::PatchMatch,
            use_bilateral: false,
            sigma_spatial: 4.0,
            sigma_color: 10.0,
            n_size_step: 1,
            extra_pass_3x3: false,
            seed: 2024,
        }
    }
}

pub struct SynthResult {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// `(height, width, channels)` stylized image.
    pub image: Vec<u8>,
    /// `(height, width)` final patch error.
    pub error: Vec<f32>,
}

/// Image size and channel count of an HWC buffer.
#[derive(Clone, Copy)]
struct Shape {
    w: u32,
    h: u32,
    c: u32,
}

/// Bilinear resize, `align_corners=False`, no antialiasing (torch `F.interpolate`),
/// rounded to uint8 by truncation like `resample_tensor`.
fn resize_u8(src: &[u8], from: Shape, w: u32, h: u32) -> Vec<u8> {
    if from.w == w && from.h == h {
        return src.to_vec();
    }
    let c = from.c as usize;
    let mut out = vec![0u8; (w * h) as usize * c];
    let xs: Vec<(usize, usize, f32)> = (0..w).map(|x| taps(x, from.w, w)).collect();
    for y in 0..h {
        let (y0, y1, ly) = taps(y, from.h, h);
        for (x, &(x0, x1, lx)) in xs.iter().enumerate() {
            for ch in 0..c {
                let at = |yy: usize, xx: usize| src[(yy * from.w as usize + xx) * c + ch] as f32;
                let v = (1.0 - ly) * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                    + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
                out[(y as usize * w as usize + x) * c + ch] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Source taps and weight of output index `o` (torch `area_pixel_compute_source_index`).
fn taps(o: u32, in_size: u32, out_size: u32) -> (usize, usize, f32) {
    let scale = in_size as f32 / out_size as f32;
    let s = ((o as f32 + 0.5) * scale - 0.5).max(0.0);
    let i0 = s as usize;
    let i1 = if i0 + 1 < in_size as usize {
        i0 + 1
    } else {
        i0
    };
    (i0, i1, s - i0 as f32)
}

/// `nnf * 2` resized bilinearly to `(w, h)`, truncated to int and clamped to the source.
fn upscale_nnf(
    nnf: &[i32],
    from_w: u32,
    from_h: u32,
    w: u32,
    h: u32,
    sw: u32,
    sh: u32,
) -> Vec<i32> {
    let mut out = vec![0i32; (w * h * 2) as usize];
    let xs: Vec<(usize, usize, f32)> = (0..w).map(|x| taps(x, from_w, w)).collect();
    for y in 0..h {
        let (y0, y1, ly) = taps(y, from_h, h);
        for (x, &(x0, x1, lx)) in xs.iter().enumerate() {
            for ch in 0..2 {
                let at =
                    |yy: usize, xx: usize| nnf[(yy * from_w as usize + xx) * 2 + ch] as f32 * 2.0;
                let v = (1.0 - ly) * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                    + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
                let hi = if ch == 0 {
                    sw as i32 - 1
                } else {
                    sh as i32 - 1
                };
                out[(y as usize * w as usize + x) * 2 + ch] = (v as i32).clamp(0, hi);
            }
        }
    }
    out
}

/// SplitMix64 for the initial random field.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u32, hi: u32) -> i32 {
        (lo as u64 + self.next() % (hi - lo) as u64) as i32
    }
}

fn random_nnf(seed: u32, tw: u32, th: u32, sw: u32, sh: u32, patch_size: u32) -> Vec<i32> {
    let r = patch_size / 2;
    let mut rng = SplitMix(seed as u64);
    let mut nnf = Vec::with_capacity((tw * th * 2) as usize);
    for _ in 0..tw * th {
        nnf.push(rng.range(r, sw - r));
        nnf.push(rng.range(r, sh - r));
    }
    nnf
}

/// `EbsynthEngine._determine_num_pyramid_levels`.
fn pyramid_levels(requested: u32, dims: [u32; 4], patch_size: u32) -> u32 {
    let min_dim = *dims.iter().min().unwrap() as f64;
    let max_levels = (0..=32u32)
        .rev()
        .find(|&l| min_dim * 2f64.powi(-(l as i32)) >= (2 * patch_size + 1) as f64)
        .map_or(0, |l| l + 1);
    requested.min(max_levels)
}

/// Stylize a `target_w x target_h` image: `style` is `(sh, sw, style_channels)` and every
/// guide pairs a style-sized source with a target-sized image. `progress(level, levels)`
/// runs after each pyramid level.
#[allow(clippy::too_many_arguments)]
pub async fn synthesize(
    synth: &mut Synthesizer,
    style: &[u8],
    style_width: u32,
    style_height: u32,
    style_channels: u32,
    guides: &[Guide<'_>],
    target_width: u32,
    target_height: u32,
    p: &SynthParams,
    progress: &mut dyn FnMut(u32, u32),
) -> Result<SynthResult, Error> {
    if guides.is_empty() {
        return Err(input_err("at least one guide is required"));
    }
    let (sw, sh, tw, th) = (style_width, style_height, target_width, target_height);
    if style.len() != (sw * sh * style_channels) as usize {
        return Err(input_err("style size does not match its dimensions"));
    }
    let cg: u32 = guides.iter().map(|g| g.channels).sum();
    let mut source_guide = vec![0u8; (sw * sh * cg) as usize];
    let mut target_guide = vec![0u8; (tw * th * cg) as usize];
    let mut guide_weights = Vec::with_capacity(cg as usize);
    let mut off = 0usize;
    for (i, g) in guides.iter().enumerate() {
        let c = g.channels as usize;
        if g.source.len() != (sw * sh) as usize * c || g.target.len() != (tw * th) as usize * c {
            return Err(input_err(format!(
                "guide {i}: images must match the style / target sizes"
            )));
        }
        for (p_, px) in g.source.chunks_exact(c).enumerate() {
            source_guide[p_ * cg as usize + off..p_ * cg as usize + off + c].copy_from_slice(px);
        }
        for (p_, px) in g.target.chunks_exact(c).enumerate() {
            target_guide[p_ * cg as usize + off..p_ * cg as usize + off + c].copy_from_slice(px);
        }
        guide_weights.extend(std::iter::repeat_n(g.weight / c as f32, c));
        off += c;
    }
    let style_weights = vec![1.0 / style_channels as f32; style_channels as usize];

    let levels = pyramid_levels(p.pyramid_levels, [sh, sw, th, tw], p.patch_size);
    if levels == 0 {
        return Err(input_err(format!(
            "images are too small for {0}x{0} patches",
            p.patch_size
        )));
    }
    let style_shape = Shape {
        w: sw,
        h: sh,
        c: style_channels,
    };
    let sg_shape = Shape {
        w: sw,
        h: sh,
        c: cg,
    };
    let tg_shape = Shape {
        w: tw,
        h: th,
        c: cg,
    };
    let level_params = |seed: u32, patch_size: u32, uniformity: f32| LevelParams {
        patch_size,
        uniformity,
        vote_mode: p.vote_mode,
        search_vote_iters: p.search_vote_iters,
        patch_match_iters: p.patch_match_iters,
        stop_threshold: p.stop_threshold,
        search_pruning_threshold: p.search_pruning_threshold,
        cost: p.cost,
        search: p.search,
        use_bilateral: p.use_bilateral,
        sigma_spatial: p.sigma_spatial,
        sigma_color: p.sigma_color,
        n_size_step: p.n_size_step,
        style_weights: style_weights.clone(),
        guide_weights: guide_weights.clone(),
        seed,
    };

    let mut nnf: Vec<i32> = Vec::new();
    let mut prev = (0u32, 0u32);
    let mut result = None;
    for level in 0..levels {
        let scale = 2f64.powi(-((levels - 1 - level) as i32));
        let lsw = ((sw as f64 * scale) as u32).max(1);
        let lsh = ((sh as f64 * scale) as u32).max(1);
        let ltw = ((tw as f64 * scale) as u32).max(1);
        let lth = ((th as f64 * scale) as u32).max(1);
        let l_style = resize_u8(style, style_shape, lsw, lsh);
        let l_sg = resize_u8(&source_guide, sg_shape, lsw, lsh);
        let l_tg = resize_u8(&target_guide, tg_shape, ltw, lth);
        nnf = if level == 0 {
            random_nnf(p.seed, ltw, lth, lsw, lsh, p.patch_size)
        } else {
            upscale_nnf(&nnf, prev.0, prev.1, ltw, lth, lsw, lsh)
        };
        let input = LevelInput {
            source_width: lsw,
            source_height: lsh,
            target_width: ltw,
            target_height: lth,
            style_channels,
            guide_channels: cg,
            source_style: &l_style,
            source_guide: &l_sg,
            target_guide: &l_tg,
            modulation: None,
            nnf: &nnf,
        };
        let out = synth
            .run_level(
                &input,
                &level_params(p.seed.wrapping_add(level), p.patch_size, p.uniformity),
            )
            .await?;
        nnf = out.nnf.clone();
        prev = (ltw, lth);
        result = Some(out);
        progress(level + 1, levels);
    }

    if p.extra_pass_3x3 {
        let input = LevelInput {
            source_width: sw,
            source_height: sh,
            target_width: tw,
            target_height: th,
            style_channels,
            guide_channels: cg,
            source_style: style,
            source_guide: &source_guide,
            target_guide: &target_guide,
            modulation: None,
            nnf: &nnf,
        };
        result = Some(
            synth
                .run_level(&input, &level_params(p.seed.wrapping_add(levels), 3, 0.0))
                .await?,
        );
    }

    let out = result.expect("at least one level ran");
    Ok(SynthResult {
        width: tw,
        height: th,
        channels: style_channels,
        image: out.image,
        error: out.error,
    })
}
