//! Browser bindings: EbSynth image synthesis on WebGPU from JavaScript.
//!
//! ```js
//! import init, { WebSynth } from "./pkg/ebsynth_web.js";
//! await init();
//! const synth = await WebSynth.create();
//! const rgba = await synth.synthesize(style, sw, sh, [srcGuide], [tgtGuide], new Float32Array([1]),
//!                                     tw, th, { patchSize: 7 }, (level, levels) => {});
//! ```
//! Images are RGBA `Uint8Array`s (canvas `ImageData.data`); alpha is ignored on input and
//! 255 on output.
//!
//! FastBlend (video blending and keyframe interpolation) shares the same device:
//!
//! ```js
//! const fb = await synth.fastBlend(height, width, { minimumPatchSize: 5, numIter: 5 });
//! // (B, H, W, 3) RGB bytes in, (B, H, W, 3) float32 remapped style out.
//! const style = await fb.estimate(batch, sourceGuides, targetGuides, sourceStyles);
//! ```

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ebsynth_gpu::fastblend::{FastBlendParams, Initialize, PyramidMatcher};
use ebsynth_gpu::{
    CostFunction, GpuContext, Guide, SearchMode, SynthParams, Synthesizer, VoteMode, synthesize,
};
use js_sys::{Array, Float32Array, Function, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect()
}

fn field(obj: &Object, key: &str) -> Option<JsValue> {
    Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null())
}

fn num(obj: &Object, key: &str) -> Result<Option<f64>, JsError> {
    field(obj, key)
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| JsError::new(&format!("option {key} must be a number")))
        })
        .transpose()
}

fn text(obj: &Object, key: &str) -> Result<Option<String>, JsError> {
    field(obj, key)
        .map(|v| {
            v.as_string()
                .ok_or_else(|| JsError::new(&format!("option {key} must be a string")))
        })
        .transpose()
}

/// `SynthParams` from a JS options object (camelCase keys; missing keys keep the defaults).
fn params(options: &JsValue) -> Result<SynthParams, JsError> {
    let mut p = SynthParams::default();
    if options.is_undefined() || options.is_null() {
        return Ok(p);
    }
    let o: &Object = options
        .dyn_ref()
        .ok_or_else(|| JsError::new("options must be an object"))?;
    if let Some(v) = num(o, "pyramidLevels")? {
        p.pyramid_levels = v as u32;
    }
    if let Some(v) = num(o, "patchSize")? {
        p.patch_size = v as u32;
    }
    if let Some(v) = num(o, "uniformity")? {
        p.uniformity = v as f32;
    }
    if let Some(v) = num(o, "searchVoteIters")? {
        p.search_vote_iters = v as u32;
    }
    if let Some(v) = num(o, "patchMatchIters")? {
        p.patch_match_iters = v as u32;
    }
    if let Some(v) = num(o, "stopThreshold")? {
        p.stop_threshold = v as i32;
    }
    if let Some(v) = num(o, "searchPruningThreshold")? {
        p.search_pruning_threshold = v as f32;
    }
    if let Some(v) = num(o, "nSizeStep")? {
        p.n_size_step = v as u32;
    }
    if let Some(v) = num(o, "sigmaSpatial")? {
        p.sigma_spatial = v as f32;
    }
    if let Some(v) = num(o, "sigmaColor")? {
        p.sigma_color = v as f32;
    }
    if let Some(v) = num(o, "seed")? {
        p.seed = v as u32;
    }
    if let Some(v) = field(o, "useBilateral") {
        p.use_bilateral = v.is_truthy();
    }
    if let Some(v) = field(o, "extraPass3x3") {
        p.extra_pass_3x3 = v.is_truthy();
    }
    if let Some(s) = text(o, "costFunction")? {
        p.cost = match s.as_str() {
            "ssd" => CostFunction::Ssd,
            "ncc" => CostFunction::Ncc,
            "sad" => CostFunction::Sad,
            _ => return Err(JsError::new(&format!("unknown costFunction {s:?}"))),
        };
    }
    if let Some(s) = text(o, "voteMode")? {
        p.vote_mode = match s.as_str() {
            "weighted" => VoteMode::Weighted,
            "plain" => VoteMode::Plain,
            _ => return Err(JsError::new(&format!("unknown voteMode {s:?}"))),
        };
    }
    let k = num(o, "multicandK")?.map_or(3, |v| v as u32);
    if let Some(s) = text(o, "searchMode")? {
        p.search = match s.as_str() {
            "patchmatch" => SearchMode::PatchMatch,
            "jfa" => SearchMode::Jfa,
            "multicand" => SearchMode::Multicand { k },
            _ => return Err(JsError::new(&format!("unknown searchMode {s:?}"))),
        };
    }
    Ok(p)
}

fn byte_arrays(a: &Array, what: &str) -> Result<Vec<Vec<u8>>, JsError> {
    a.iter()
        .map(|v| {
            v.dyn_into::<Uint8Array>()
                .map(|u| u.to_vec())
                .map_err(|_| JsError::new(&format!("{what} must be Uint8Arrays")))
        })
        .collect()
}

/// WebGPU EbSynth synthesizer (one device, cached pipelines).
#[wasm_bindgen]
pub struct WebSynth {
    /// Taken out while a synthesis runs, so overlapping calls fail instead of sharing it.
    inner: Rc<RefCell<Option<Synthesizer>>>,
    ctx: Arc<GpuContext>,
    adapter: String,
}

#[wasm_bindgen]
impl WebSynth {
    /// Request a WebGPU adapter and device.
    pub async fn create() -> Result<WebSynth, JsError> {
        let ctx = GpuContext::new()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let adapter = ctx.adapter_summary();
        // The browser is single-threaded; the core API takes an Arc for native callers.
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx = Arc::new(ctx);
        Ok(WebSynth {
            inner: Rc::new(RefCell::new(Some(Synthesizer::new(ctx.clone())))),
            ctx,
            adapter,
        })
    }

    /// Adapter name and backend.
    #[wasm_bindgen(getter)]
    pub fn adapter(&self) -> String {
        self.adapter.clone()
    }

    /// Largest storage-buffer binding the device allows, in bytes (bounds batch sizes).
    #[wasm_bindgen(getter, js_name = maxStorageBinding)]
    pub fn max_storage_binding(&self) -> f64 {
        self.ctx.limits.max_storage_buffer_binding_size as f64
    }

    /// Build a FastBlend pyramid matcher for `height x width` RGB frames on this device.
    /// Resolves to a [`WebFastBlend`]. Options (camelCase, all optional): `minimumPatchSize`
    /// (odd, 5), `numIter` (5), `guideWeight` (10), `useMeanTargetStyle` (false),
    /// `usePairwisePatchError` (false), `trackingWindowSize` (0), `initialize`
    /// ("identity" | "random"), `seed` (0).
    #[wasm_bindgen(js_name = fastBlend)]
    pub fn fast_blend(
        &self,
        height: u32,
        width: u32,
        options: JsValue,
    ) -> Result<js_sys::Promise, JsError> {
        let p = fastblend_params(height, width, &options)?;
        let ctx = self.ctx.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let matcher = PyramidMatcher::new(ctx, p)
                .await
                .map_err(|e| JsError::new(&e.to_string()))?;
            let levels = matcher.levels() as u32;
            Ok(WebFastBlend {
                inner: Rc::new(RefCell::new(Some(matcher))),
                height,
                width,
                levels,
            }
            .into())
        }))
    }

    /// Stylize a `target_width x target_height` image. `style` and each source guide are
    /// `style_width x style_height` RGBA; target guides are target-sized RGBA. `progress`
    /// is called with `(level, levels)`. Resolves to the RGBA result.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &self,
        style: Vec<u8>,
        style_width: u32,
        style_height: u32,
        source_guides: Array,
        target_guides: Array,
        weights: Vec<f32>,
        target_width: u32,
        target_height: u32,
        options: JsValue,
        progress: Option<Function>,
    ) -> Result<js_sys::Promise, JsError> {
        self.run(
            style,
            (style_width, style_height),
            source_guides,
            target_guides,
            weights,
            (target_width, target_height),
            options,
            progress,
            false,
        )
    }

    /// Like `synthesize`, but resolves to `{ image, error }`: the RGBA result and its
    /// per-pixel patch error (`Float32Array`, target_width x target_height), which the video
    /// pipeline uses to choose between forward and backward passes.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = synthesizeWithError)]
    pub fn synthesize_with_error(
        &self,
        style: Vec<u8>,
        style_width: u32,
        style_height: u32,
        source_guides: Array,
        target_guides: Array,
        weights: Vec<f32>,
        target_width: u32,
        target_height: u32,
        options: JsValue,
        progress: Option<Function>,
    ) -> Result<js_sys::Promise, JsError> {
        self.run(
            style,
            (style_width, style_height),
            source_guides,
            target_guides,
            weights,
            (target_width, target_height),
            options,
            progress,
            true,
        )
    }
}

impl WebSynth {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        style: Vec<u8>,
        (style_width, style_height): (u32, u32),
        source_guides: Array,
        target_guides: Array,
        weights: Vec<f32>,
        (target_width, target_height): (u32, u32),
        options: JsValue,
        progress: Option<Function>,
        with_error: bool,
    ) -> Result<js_sys::Promise, JsError> {
        let p = params(&options)?;
        let sources = byte_arrays(&source_guides, "source guides")?;
        let targets = byte_arrays(&target_guides, "target guides")?;
        if sources.len() != targets.len() || sources.len() != weights.len() {
            return Err(JsError::new(
                "source guides, target guides and weights must have equal lengths",
            ));
        }
        let inner = self.inner.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let style_rgb = rgba_to_rgb(&style);
            let src_rgb: Vec<Vec<u8>> = sources.iter().map(|s| rgba_to_rgb(s)).collect();
            let tgt_rgb: Vec<Vec<u8>> = targets.iter().map(|t| rgba_to_rgb(t)).collect();
            let guides: Vec<Guide> = src_rgb
                .iter()
                .zip(&tgt_rgb)
                .zip(&weights)
                .map(|((s, t), &w)| Guide {
                    source: s,
                    target: t,
                    channels: 3,
                    weight: w,
                })
                .collect();
            let mut report = |level: u32, levels: u32| {
                if let Some(f) = &progress {
                    let _ = f.call2(&JsValue::NULL, &level.into(), &levels.into());
                }
            };
            let mut synth = inner
                .borrow_mut()
                .take()
                .ok_or_else(|| JsError::new("a synthesis is already running"))?;
            let out = synthesize(
                &mut synth,
                &style_rgb,
                style_width,
                style_height,
                3,
                &guides,
                target_width,
                target_height,
                &p,
                &mut report,
            )
            .await;
            *inner.borrow_mut() = Some(synth);
            let out = out.map_err(|e| JsError::new(&e.to_string()))?;
            let image: JsValue = Uint8Array::from(rgb_to_rgba(&out.image).as_slice()).into();
            if !with_error {
                return Ok(image);
            }
            let result = Object::new();
            Reflect::set(&result, &"image".into(), &image).map_err(|_| JsError::new("result"))?;
            Reflect::set(
                &result,
                &"error".into(),
                &Float32Array::from(out.error.as_slice()).into(),
            )
            .map_err(|_| JsError::new("result"))?;
            Ok(result.into())
        }))
    }
}

/// `FastBlendParams` for 3-channel frames from a JS options object.
fn fastblend_params(
    height: u32,
    width: u32,
    options: &JsValue,
) -> Result<FastBlendParams, JsError> {
    let mut p = FastBlendParams {
        height,
        width,
        channels: 3,
        minimum_patch_size: 5,
        num_iter: 5,
        guide_weight: 10.0,
        use_mean_target_style: false,
        use_pairwise_patch_error: false,
        tracking_window_size: 0,
        initialize: Initialize::Identity,
        random_search_steps: 3,
        random_search_range: 4,
        seed: 0,
    };
    if options.is_undefined() || options.is_null() {
        return Ok(p);
    }
    let o: &Object = options
        .dyn_ref()
        .ok_or_else(|| JsError::new("options must be an object"))?;
    if let Some(v) = num(o, "minimumPatchSize")? {
        p.minimum_patch_size = v as u32;
    }
    if let Some(v) = num(o, "numIter")? {
        p.num_iter = v as u32;
    }
    if let Some(v) = num(o, "guideWeight")? {
        p.guide_weight = v as f32;
    }
    if let Some(v) = num(o, "trackingWindowSize")? {
        p.tracking_window_size = v as u32;
    }
    if let Some(v) = num(o, "seed")? {
        p.seed = v as u64;
    }
    if let Some(v) = field(o, "useMeanTargetStyle") {
        p.use_mean_target_style = v.is_truthy();
    }
    if let Some(v) = field(o, "usePairwisePatchError") {
        p.use_pairwise_patch_error = v.is_truthy();
    }
    if let Some(s) = text(o, "initialize")? {
        p.initialize = match s.as_str() {
            "identity" => Initialize::Identity,
            "random" => Initialize::Random,
            _ => return Err(JsError::new(&format!("unknown initialize {s:?}"))),
        };
    }
    Ok(p)
}

/// FastBlend pyramid patch matcher on a [`WebSynth`] device (`PyramidPatchMatcherCore`
/// semantics); the blending and interpolation runners live in JavaScript.
#[wasm_bindgen]
pub struct WebFastBlend {
    /// Taken out while a batch runs, so overlapping calls fail instead of sharing it.
    inner: Rc<RefCell<Option<PyramidMatcher>>>,
    height: u32,
    width: u32,
    levels: u32,
}

#[wasm_bindgen]
impl WebFastBlend {
    /// Pyramid levels used for this frame size.
    #[wasm_bindgen(getter)]
    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Remap `batch` source styles onto their target guides. Each input is `batch` frames
    /// of `height x width` RGB bytes, concatenated. Resolves to a `Float32Array` of
    /// `batch x height x width x 3` remapped style values (0..255).
    pub fn estimate(
        &self,
        batch: u32,
        source_guide: Vec<u8>,
        target_guide: Vec<u8>,
        source_style: Vec<u8>,
    ) -> Result<js_sys::Promise, JsError> {
        let full = (batch * self.height * self.width * 3) as usize;
        for (name, len) in [
            ("source guides", source_guide.len()),
            ("target guides", target_guide.len()),
            ("source styles", source_style.len()),
        ] {
            if len != full {
                return Err(JsError::new(&format!(
                    "{name} have {len} bytes, expected {batch}x{}x{}x3",
                    self.height, self.width
                )));
            }
        }
        let inner = self.inner.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let floats = |b: &[u8]| b.iter().map(|&v| f32::from(v)).collect::<Vec<f32>>();
            let (sg, tg, ss) = (
                floats(&source_guide),
                floats(&target_guide),
                floats(&source_style),
            );
            drop((source_guide, target_guide, source_style));
            let mut matcher = inner
                .borrow_mut()
                .take()
                .ok_or_else(|| JsError::new("a FastBlend batch is already running"))?;
            let out = matcher.estimate_nnf(batch, &sg, &tg, &ss).await;
            *inner.borrow_mut() = Some(matcher);
            let out = out.map_err(|e| JsError::new(&e.to_string()))?;
            Ok(Float32Array::from(out.target_style.as_slice()).into())
        }))
    }
}
