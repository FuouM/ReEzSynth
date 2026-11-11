//! Buffer, readback and error-scope helpers shared by the synthesizers.

use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::error::Error;

pub(crate) const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

pub(crate) fn storage_init(ctx: &GpuContext, label: &str, data: &[u8]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: STORAGE,
        })
}

/// Zero-initialized storage buffer (WebGPU zero-fills new buffers).
pub(crate) fn storage_zeroed(ctx: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4).next_multiple_of(4),
        usage: STORAGE,
        mapped_at_creation: false,
    })
}

pub(crate) fn uniform_init(ctx: &GpuContext, label: &str, data: &[u8]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(
    binding: u32,
    dynamic: bool,
    min_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(min_size),
        },
        count: None,
    }
}

/// Check a buffer size against the device's storage binding limit.
pub(crate) fn check_binding_size(ctx: &GpuContext, what: &str, bytes: u64) -> Result<(), Error> {
    let limit = ctx.limits.max_storage_buffer_binding_size;
    if bytes > limit {
        return Err(Error::Input(format!(
            "{what} needs a {} MiB buffer; this adapter allows {} MiB per binding",
            bytes >> 20,
            limit >> 20
        )));
    }
    Ok(())
}

/// Wait for queued GPU work (native); the browser drives the queue itself.
pub(crate) fn wait_idle(ctx: &GpuContext) -> Result<(), Error> {
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| Error::Gpu(format!("poll failed: {e}")))?;
    #[cfg(target_arch = "wasm32")]
    let _ = ctx;
    Ok(())
}

/// Copy `(buffer, offset, size)` ranges to the host in one submission.
pub(crate) async fn read_ranges(
    ctx: &GpuContext,
    ranges: &[(&wgpu::Buffer, u64, u64)],
) -> Result<Vec<Vec<u8>>, Error> {
    let total: u64 = ranges.iter().map(|r| r.2).sum();
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: total.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    let mut at = 0;
    for (buf, off, size) in ranges {
        enc.copy_buffer_to_buffer(buf, *off, &staging, at, *size);
        at += size;
    }
    ctx.queue.submit([enc.finish()]);
    let (tx, rx) = futures_channel::oneshot::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    wait_idle(ctx)?;
    rx.await
        .map_err(|_| Error::Gpu("readback callback dropped".into()))?
        .map_err(|e| Error::Gpu(format!("buffer map failed: {e}")))?;
    let view = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|e| Error::Gpu(format!("mapped range: {e:?}")))?;
    let mut out = Vec::with_capacity(ranges.len());
    let mut at = 0usize;
    for (_, _, size) in ranges {
        out.push(view[at..at + *size as usize].to_vec());
        at += *size as usize;
    }
    drop(view);
    staging.unmap();
    Ok(out)
}

/// Run `f` inside validation + out-of-memory error scopes and report captured errors.
pub(crate) async fn checked<T>(ctx: &GpuContext, f: impl FnOnce() -> T) -> Result<T, Error> {
    let oom = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    let v = validation.pop().await;
    let o = oom.pop().await;
    let earlier = ctx.take_failure();
    match (v.or(o), earlier) {
        (None, None) => Ok(out),
        (Some(e), None) => Err(Error::Gpu(e.to_string())),
        (None, Some(msg)) => Err(Error::Gpu(msg)),
        (Some(e), Some(msg)) => Err(Error::Gpu(format!("{e}\n(earlier: {msg})"))),
    }
}
