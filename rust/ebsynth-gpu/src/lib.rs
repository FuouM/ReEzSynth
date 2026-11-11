//! WebGPU implementation of ReEzSynth's EbSynth level synthesis and FastBlend patch
//! matching. The same code runs natively (Vulkan, DX12, Metal) and in the browser
//! (wasm32 + WebGPU); every entry point is async so the browser never blocks.

mod context;
mod ebsynth;
mod error;
pub mod fastblend;
mod gpu;
mod pyramid;

pub use context::GpuContext;
pub use ebsynth::{
    CostFunction, LevelInput, LevelOutput, LevelParams, MAX_GUIDE_CHANNELS, MAX_STYLE_CHANNELS,
    SearchMode, Synthesizer, VoteMode,
};
pub use error::Error;
pub use pyramid::{Guide, SynthParams, SynthResult, synthesize};
