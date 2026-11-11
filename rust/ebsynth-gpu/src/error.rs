use std::fmt;

/// Errors from device setup, input validation and GPU execution.
#[derive(Debug)]
pub enum Error {
    /// No WebGPU adapter, or the device request failed.
    Device(String),
    /// Inputs do not match the documented shapes or limits.
    Input(String),
    /// The GPU reported a validation or out-of-memory error, or the device was lost.
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Device(m) => write!(f, "WebGPU device error: {m}"),
            Error::Input(m) => write!(f, "invalid input: {m}"),
            Error::Gpu(m) => write!(f, "GPU error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn input_err(msg: impl Into<String>) -> Error {
    Error::Input(msg.into())
}
