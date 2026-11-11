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

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ebsynth_gpu::{
    CostFunction, GpuContext, Guide, SearchMode, SynthParams, Synthesizer, VoteMode, synthesize,
};
use js_sys::{Array, Function, Object, Reflect, Uint8Array};
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
            inner: Rc::new(RefCell::new(Some(Synthesizer::new(ctx)))),
            adapter,
        })
    }

    /// Adapter name and backend.
    #[wasm_bindgen(getter)]
    pub fn adapter(&self) -> String {
        self.adapter.clone()
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
            Ok(Uint8Array::from(rgb_to_rgba(&out.image).as_slice()).into())
        }))
    }
}
