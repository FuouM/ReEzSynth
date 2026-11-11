//! Python module `ezsynth_wgpu`: the WebGPU synthesis backend of ReEzSynth.

use std::sync::{Arc, LazyLock, Mutex};

use ebsynth_gpu::fastblend::{FastBlendParams, Initialize, PyramidMatcher};
use ebsynth_gpu::{
    CostFunction, Error, GpuContext, LevelInput, LevelParams, SearchMode,
    Synthesizer as CoreSynthesizer, VoteMode,
};
use numpy::ndarray::Dim;
use numpy::{
    IntoPyArray, PyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
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
    m.add("MAX_STYLE_CHANNELS", ebsynth_gpu::MAX_STYLE_CHANNELS)?;
    m.add("MAX_GUIDE_CHANNELS", ebsynth_gpu::MAX_GUIDE_CHANNELS)?;
    Ok(())
}
