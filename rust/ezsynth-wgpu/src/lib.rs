//! Python module `ezsynth_wgpu`: the WebGPU synthesis backend of ReEzSynth.

use std::sync::{Arc, LazyLock, Mutex};

use ebsynth_gpu::fastblend::{FastBlendParams, Initialize, PyramidMatcher};
use ebsynth_gpu::imageops::{Border, ImageOps, Interpolation, OcclusionParams};
use ebsynth_gpu::{
    CostFunction, Error, GpuContext, LevelInput, LevelParams, SearchMode,
    Synthesizer as CoreSynthesizer, VoteMode,
};
use numpy::ndarray::{Dim, IxDyn};
use numpy::{
    IntoPyArray, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// `run_level` result: image (th, tw, cs) uint8, error (th, tw) float32, nnf (th, tw, 2) int32.
type LevelArrays<'py> = (
    Bound<'py, PyArray<u8, Dim<[usize; 3]>>>,
    Bound<'py, PyArray<f32, Dim<[usize; 2]>>>,
    Bound<'py, PyArray<i32, Dim<[usize; 3]>>>,
);

/// `estimate_nnf` result: nnf (B, H, W, 2) int32, target style (B, H, W, C) float32.
type MatchArrays<'py> = (
    Bound<'py, PyArray<i32, Dim<[usize; 4]>>>,
    Bound<'py, PyArray<f32, Dim<[usize; 4]>>>,
);

static CONTEXT: LazyLock<Result<Arc<GpuContext>, String>> = LazyLock::new(|| {
    pollster::block_on(GpuContext::new())
        .map(Arc::new)
        .map_err(|e| e.to_string())
});

fn context() -> PyResult<Arc<GpuContext>> {
    CONTEXT.clone().map_err(PyRuntimeError::new_err)
}

static IMAGE_OPS: LazyLock<Result<ImageOps, String>> = LazyLock::new(|| {
    let ctx = CONTEXT.clone()?;
    pollster::block_on(ImageOps::new(ctx)).map_err(|e| e.to_string())
});

fn image_ops() -> PyResult<&'static ImageOps> {
    IMAGE_OPS
        .as_ref()
        .map_err(|e| PyRuntimeError::new_err(e.clone()))
}

/// Data and (h, w, c) of a float32 `(H, W)` or `(H, W, C)` image.
fn image_f32<'a>(
    name: &str,
    a: &'a PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(&'a [f32], u32, u32, u32)> {
    let s = a.shape();
    let c = match s.len() {
        2 => 1,
        3 => s[2],
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must be (H, W) or (H, W, C), got shape {s:?}"
            )));
        }
    };
    Ok((slice(name, a)?, s[0] as u32, s[1] as u32, c as u32))
}

/// Flow `(h, w, 2)` float32 matching an image.
fn flow_f32<'a>(
    name: &str,
    a: &'a PyReadonlyArray3<'_, f32>,
    h: u32,
    w: u32,
) -> PyResult<&'a [f32]> {
    if a.shape() != [h as usize, w as usize, 2] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({h}, {w}, 2), got {:?}",
            a.shape()
        )));
    }
    slice(name, a)
}

/// Image (shape of the input) plus a `(H, W)` float32 map.
type ImageAndMap<'py> = (Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<f32>>);

/// Two `(H, W)` uint8 masks.
type MaskPair<'py> = (
    Bound<'py, PyArray<u8, Dim<[usize; 2]>>>,
    Bound<'py, PyArray<u8, Dim<[usize; 2]>>>,
);

fn dyn_array<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    data.into_pyarray(py).reshape(IxDyn(shape))
}

/// `cv2.remap(src, grid + flow)` with OpenCV 5's bilinear (exact float coordinates for
/// 1, 3 and 4 channels, 1/32-pixel fixed point otherwise) or nearest sampling. `src` is
/// float32 `(H, W)` or `(H, W, C)`, `flow` float32 `(H, W, 2)`; `border` is "reflect"
/// (`BORDER_REFLECT`) or "constant" (0).
#[pyfunction]
#[pyo3(signature = (src, flow, *, nearest=false, border="reflect"))]
fn warp<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    flow: PyReadonlyArray3<'py, f32>,
    nearest: bool,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let border = match border {
        "reflect" => Border::Reflect,
        "constant" => Border::Constant,
        _ => {
            return Err(PyValueError::new_err(format!(
                "border must be 'reflect' or 'constant', got {border:?}"
            )));
        }
    };
    let interp = if nearest {
        Interpolation::Nearest
    } else {
        Interpolation::Linear
    };
    let (data, h, w, c) = image_f32("src", &src)?;
    let flow = flow_f32("flow", &flow, h, w)?;
    let ops = image_ops()?;
    let out = py
        .detach(|| pollster::block_on(ops.warp(data, h, w, c, flow, interp, border)))
        .map_err(to_py)?;
    dyn_array(py, out, src.shape())
}

/// Forward soft splatting of float32 `src` `(H, W[, C])` along `flow` `(H, W, 2)` with
/// optional bilateral `(H, W, 3)` guides and pull-push hole filling. Returns
/// `(image float32 like src, splatted weight (H, W) before hole filling)`.
#[pyfunction]
#[pyo3(signature = (src, flow, *, fill_holes=true, src_guide=None, tgt_guide=None))]
fn forward_splat<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    flow: PyReadonlyArray3<'py, f32>,
    fill_holes: bool,
    src_guide: Option<PyReadonlyArray3<'py, f32>>,
    tgt_guide: Option<PyReadonlyArray3<'py, f32>>,
) -> PyResult<ImageAndMap<'py>> {
    let (data, h, w, c) = image_f32("src", &src)?;
    let flow = flow_f32("flow", &flow, h, w)?;
    let guides = match (&src_guide, &tgt_guide) {
        (Some(sg), Some(tg)) => Some((slice("src_guide", sg)?, slice("tgt_guide", tg)?)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "src_guide and tgt_guide must be given together",
            ));
        }
    };
    let ops = image_ops()?;
    let out = py
        .detach(|| pollster::block_on(ops.forward_splat(data, h, w, c, flow, guides, fill_holes)))
        .map_err(to_py)?;
    Ok((
        dyn_array(py, out.image, src.shape())?,
        dyn_array(py, out.weight, &[h as usize, w as usize])?,
    ))
}

/// Pull-push hole filling of accumulated float32 `color` `(H, W[, C])` and `weight`
/// `(H, W)`; returns the filled `(color, weight)`.
#[pyfunction]
#[pyo3(signature = (color, weight, *, levels=5))]
fn pull_push<'py>(
    py: Python<'py>,
    color: PyReadonlyArrayDyn<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    levels: u32,
) -> PyResult<ImageAndMap<'py>> {
    let (data, h, w, c) = image_f32("color", &color)?;
    if weight.shape() != [h as usize, w as usize] {
        return Err(PyValueError::new_err(format!(
            "weight must have shape ({h}, {w}), got {:?}",
            weight.shape()
        )));
    }
    let wdata = slice("weight", &weight)?;
    let ops = image_ops()?;
    let (color_out, weight_out) = py
        .detach(|| pollster::block_on(ops.pull_push(data, wdata, h, w, c, levels)))
        .map_err(to_py)?;
    Ok((
        dyn_array(py, color_out, color.shape())?,
        dyn_array(py, weight_out, &[h as usize, w as usize])?,
    ))
}

/// Histogram blend of uint8 BGR frames `a`, `b` `(H, W, 3)` with the selection `mask`
/// `(H, W)` uint8 (0 where `a` has the lower error). Returns BGR uint8 `(H, W, 3)`.
#[pyfunction]
#[pyo3(signature = (a, b, mask, weight1=0.5, weight2=0.5))]
fn hist_blend<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, u8>,
    b: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray2<'py, u8>,
    weight1: f32,
    weight2: f32,
) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 3]>>>> {
    let (w, h, c) = shape3(&a);
    if c != 3 || b.shape() != a.shape() || mask.shape() != [h as usize, w as usize] {
        return Err(PyValueError::new_err(format!(
            "a and b must be (H, W, 3) and mask (H, W); got {:?}, {:?}, {:?}",
            a.shape(),
            b.shape(),
            mask.shape()
        )));
    }
    let (ad, bd, md) = (slice("a", &a)?, slice("b", &b)?, slice("mask", &mask)?);
    let ops = image_ops()?;
    let out = py
        .detach(|| pollster::block_on(ops.hist_blend(ad, bd, md, h, w, weight1, weight2)))
        .map_err(to_py)?;
    out.into_pyarray(py).reshape([h as usize, w as usize, 3])
}

/// Unreliable-pixel masks (uint8 0 / 255) of one flow pair: `fwd_flow` maps frame i to
/// i + 1, `bwd_flow` i + 1 to i, both float32 `(H, W, 2)`. Returns `(fwd_mask (pixels
/// of frame i + 1), bwd_mask (pixels of frame i))`.
#[pyfunction]
#[pyo3(signature = (fwd_flow, bwd_flow, *, alpha=0.01, beta=0.5, dilate=1, coverage=true))]
fn occlusion_masks<'py>(
    py: Python<'py>,
    fwd_flow: PyReadonlyArray3<'py, f32>,
    bwd_flow: PyReadonlyArray3<'py, f32>,
    alpha: f32,
    beta: f32,
    dilate: u32,
    coverage: bool,
) -> PyResult<MaskPair<'py>> {
    let s = fwd_flow.shape();
    let (h, w) = (s[0] as u32, s[1] as u32);
    let fwd = flow_f32("fwd_flow", &fwd_flow, h, w)?;
    let bwd = flow_f32("bwd_flow", &bwd_flow, h, w)?;
    let params = OcclusionParams {
        alpha,
        beta,
        dilate,
        coverage,
    };
    let ops = image_ops()?;
    let (fm, bm) = py
        .detach(|| pollster::block_on(ops.occlusion_masks(fwd, bwd, h, w, params)))
        .map_err(to_py)?;
    let shape = [h as usize, w as usize];
    Ok((
        fm.into_pyarray(py).reshape(shape)?,
        bm.into_pyarray(py).reshape(shape)?,
    ))
}

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Input(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn slice<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    name: &str,
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous array")))
}

/// (width, height, channels) of an HWC array.
fn shape3(a: &PyReadonlyArray3<'_, u8>) -> (u32, u32, u32) {
    let s = a.shape();
    (s[1] as u32, s[0] as u32, s[2] as u32)
}

/// Adapter name and graphics API of the shared WebGPU device, e.g. "NVIDIA ... (Vulkan)".
#[pyfunction]
fn adapter_info() -> PyResult<String> {
    Ok(context()?.adapter_summary())
}

fn parse_cost(s: &str) -> PyResult<CostFunction> {
    match s {
        "ssd" => Ok(CostFunction::Ssd),
        "ncc" => Ok(CostFunction::Ncc),
        "sad" => Ok(CostFunction::Sad),
        _ => Err(PyValueError::new_err(format!(
            "unknown cost_function {s:?}; expected ssd, ncc or sad"
        ))),
    }
}

fn parse_search(s: &str, k: u32) -> PyResult<SearchMode> {
    match s {
        "patchmatch" => Ok(SearchMode::PatchMatch),
        "jfa" => Ok(SearchMode::Jfa),
        "multicand" => Ok(SearchMode::Multicand { k }),
        _ => Err(PyValueError::new_err(format!(
            "unknown search_mode {s:?}; expected patchmatch, jfa or multicand"
        ))),
    }
}

fn parse_vote(s: &str) -> PyResult<VoteMode> {
    match s {
        "weighted" => Ok(VoteMode::Weighted),
        "plain" => Ok(VoteMode::Plain),
        _ => Err(PyValueError::new_err(format!(
            "unknown vote_mode {s:?}; expected weighted or plain"
        ))),
    }
}

/// EbSynth level synthesizer on the shared WebGPU device.
#[pyclass]
struct Synthesizer {
    inner: Mutex<CoreSynthesizer>,
}

#[pymethods]
impl Synthesizer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Synthesizer {
            inner: Mutex::new(CoreSynthesizer::new(context()?)),
        })
    }

    #[getter]
    fn adapter(&self) -> String {
        self.inner.lock().unwrap().context().adapter_summary()
    }

    /// Run one pyramid level. Images are HWC uint8; `nnf` is `(th, tw, 2)` int32 source
    /// `(x, y)` centres. Returns `(image (th, tw, cs) uint8, error (th, tw) float32,
    /// nnf (th, tw, 2) int32)`.
    #[pyo3(signature = (
        source_style, source_guide, target_guide, modulation, nnf, style_weights, guide_weights, *,
        patch_size, uniformity, vote_mode, search_vote_iters, patch_match_iters, stop_threshold,
        search_pruning_threshold, cost_function, search_mode, multicand_k, use_bilateral,
        sigma_spatial, sigma_color, n_size_step, seed
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_level<'py>(
        &self,
        py: Python<'py>,
        source_style: PyReadonlyArray3<'py, u8>,
        source_guide: PyReadonlyArray3<'py, u8>,
        target_guide: PyReadonlyArray3<'py, u8>,
        modulation: Option<PyReadonlyArray3<'py, u8>>,
        nnf: PyReadonlyArray3<'py, i32>,
        style_weights: PyReadonlyArray1<'py, f32>,
        guide_weights: PyReadonlyArray1<'py, f32>,
        patch_size: u32,
        uniformity: f32,
        vote_mode: &str,
        search_vote_iters: u32,
        patch_match_iters: u32,
        stop_threshold: i32,
        search_pruning_threshold: f32,
        cost_function: &str,
        search_mode: &str,
        multicand_k: u32,
        use_bilateral: bool,
        sigma_spatial: f32,
        sigma_color: f32,
        n_size_step: u32,
        seed: u32,
    ) -> PyResult<LevelArrays<'py>> {
        let (sw, sh, cs) = shape3(&source_style);
        let (sgw, sgh, cg) = shape3(&source_guide);
        let (tw, th, tcg) = shape3(&target_guide);
        if (sgw, sgh) != (sw, sh) {
            return Err(PyValueError::new_err(
                "source_guide must have the style's height and width",
            ));
        }
        if tcg != cg {
            return Err(PyValueError::new_err(
                "source_guide and target_guide channel counts differ",
            ));
        }
        let params = LevelParams {
            patch_size,
            uniformity,
            vote_mode: parse_vote(vote_mode)?,
            search_vote_iters,
            patch_match_iters,
            stop_threshold,
            search_pruning_threshold,
            cost: parse_cost(cost_function)?,
            search: parse_search(search_mode, multicand_k)?,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
            style_weights: slice("style_weights", &style_weights)?.to_vec(),
            guide_weights: slice("guide_weights", &guide_weights)?.to_vec(),
            seed,
        };
        let input = LevelInput {
            source_width: sw,
            source_height: sh,
            target_width: tw,
            target_height: th,
            style_channels: cs,
            guide_channels: cg,
            source_style: slice("source_style", &source_style)?,
            source_guide: slice("source_guide", &source_guide)?,
            target_guide: slice("target_guide", &target_guide)?,
            modulation: modulation
                .as_ref()
                .map(|m| slice("modulation", m))
                .transpose()?,
            nnf: slice("nnf", &nnf)?,
        };
        let inner = &self.inner;
        let out = py
            .detach(|| pollster::block_on(inner.lock().unwrap().run_level(&input, &params)))
            .map_err(to_py)?;
        let (h, w) = (th as usize, tw as usize);
        Ok((
            out.image.into_pyarray(py).reshape([h, w, cs as usize])?,
            out.error.into_pyarray(py).reshape([h, w])?,
            out.nnf.into_pyarray(py).reshape([h, w, 2])?,
        ))
    }
}

/// FastBlend pyramid patch matcher (`PyramidPatchMatcherCore` semantics) on the shared device.
#[pyclass]
struct FastBlendMatcher {
    inner: Mutex<PyramidMatcher>,
}

#[pymethods]
impl FastBlendMatcher {
    #[new]
    #[pyo3(signature = (
        image_height, image_width, channel, minimum_patch_size, *, num_iter, guide_weight,
        use_mean_target_style, use_pairwise_patch_error, tracking_window_size, initialize,
        random_search_steps=3, random_search_range=4, seed=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        image_height: u32,
        image_width: u32,
        channel: u32,
        minimum_patch_size: u32,
        num_iter: u32,
        guide_weight: f32,
        use_mean_target_style: bool,
        use_pairwise_patch_error: bool,
        tracking_window_size: u32,
        initialize: &str,
        random_search_steps: u32,
        random_search_range: i32,
        seed: u64,
    ) -> PyResult<Self> {
        let initialize = match initialize {
            "identity" => Initialize::Identity,
            "random" => Initialize::Random,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown initialize {initialize:?}"
                )));
            }
        };
        let params = FastBlendParams {
            height: image_height,
            width: image_width,
            channels: channel,
            minimum_patch_size,
            num_iter,
            guide_weight,
            use_mean_target_style,
            use_pairwise_patch_error,
            tracking_window_size,
            initialize,
            random_search_steps,
            random_search_range,
            seed,
        };
        let m = pollster::block_on(PyramidMatcher::new(context()?, params)).map_err(to_py)?;
        Ok(FastBlendMatcher {
            inner: Mutex::new(m),
        })
    }

    /// Pyramid levels used for this image size.
    #[getter]
    fn pyramid_level(&self) -> usize {
        self.inner.lock().unwrap().levels()
    }

    /// Match a batch: inputs are `(B, H, W, C)` float32. Returns `(nnf (B, H, W, 2) int32
    /// with (row, col) source coordinates, target_style (B, H, W, C) float32)`.
    fn estimate_nnf<'py>(
        &self,
        py: Python<'py>,
        source_guide: PyReadonlyArray4<'py, f32>,
        target_guide: PyReadonlyArray4<'py, f32>,
        source_style: PyReadonlyArray4<'py, f32>,
    ) -> PyResult<MatchArrays<'py>> {
        let s = source_guide.shape().to_vec();
        if target_guide.shape() != s.as_slice() || source_style.shape()[..3] != s[..3] {
            return Err(PyValueError::new_err(
                "source_guide, target_guide and source_style shapes must match",
            ));
        }
        let batch = s[0] as u32;
        let sg = slice("source_guide", &source_guide)?;
        let tg = slice("target_guide", &target_guide)?;
        let ss = slice("source_style", &source_style)?;
        let inner = &self.inner;
        let out = py
            .detach(|| pollster::block_on(inner.lock().unwrap().estimate_nnf(batch, sg, tg, ss)))
            .map_err(to_py)?;
        let (b, h, w) = (s[0], s[1], s[2]);
        let c = source_style.shape()[3];
        Ok((
            out.nnf.into_pyarray(py).reshape([b, h, w, 2])?,
            out.target_style.into_pyarray(py).reshape([b, h, w, c])?,
        ))
    }
}

#[pymodule]
fn ezsynth_wgpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adapter_info, m)?)?;
    m.add_class::<Synthesizer>()?;
    m.add_class::<FastBlendMatcher>()?;
    m.add_function(wrap_pyfunction!(warp, m)?)?;
    m.add_function(wrap_pyfunction!(forward_splat, m)?)?;
    m.add_function(wrap_pyfunction!(pull_push, m)?)?;
    m.add_function(wrap_pyfunction!(hist_blend, m)?)?;
    m.add_function(wrap_pyfunction!(occlusion_masks, m)?)?;
    m.add("MAX_STYLE_CHANNELS", ebsynth_gpu::MAX_STYLE_CHANNELS)?;
    m.add("MAX_GUIDE_CHANNELS", ebsynth_gpu::MAX_GUIDE_CHANNELS)?;
    Ok(())
}
