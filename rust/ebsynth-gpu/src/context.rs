use std::sync::{Arc, Mutex};

use crate::error::Error;

/// A WebGPU device shared by the synthesizers.
///
/// Native builds honour wgpu's environment variables (`WGPU_BACKEND`, `WGPU_ADAPTER_NAME`,
/// `WGPU_POWER_PREF`); otherwise the high-performance adapter of the primary backends
/// (Vulkan, DX12, Metal, browser WebGPU) is used.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub(crate) failure: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Error> {
        let desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        }
        .with_env();
        let instance = wgpu::Instance::new(desc);
        let adapter = if std::env::var_os("WGPU_ADAPTER_NAME").is_some() {
            wgpu::util::initialize_adapter_from_env(&instance, None).await
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::from_env()
                        .unwrap_or(wgpu::PowerPreference::HighPerformance),
                    ..Default::default()
                })
                .await
        }
        .map_err(|e| Error::Device(format!("no WebGPU adapter: {e}")))?;

        let al = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ebsynth-gpu"),
                required_limits: required_limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Device(format!("request_device failed: {e}")))?;

        let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let on_error = failure.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = on_error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }));
        let on_lost = failure.clone();
        device.set_device_lost_callback(move |reason, message| {
            let mut slot = on_lost.lock().unwrap();
            if slot.is_none() {
                *slot = Some(format!("device lost ({reason:?}): {message}"));
            }
        });

        Ok(GpuContext {
            device,
            queue,
            info: adapter.get_info(),
            limits: required_limits,
            failure,
        })
    }

    /// "<adapter name> (<backend>)"; browsers may hide the name, then the driver is shown.
    pub fn adapter_summary(&self) -> String {
        let i = &self.info;
        let name = [&i.name, &i.driver, &i.driver_info]
            .into_iter()
            .find(|s| !s.is_empty())
            .map_or("WebGPU adapter", |s| s.as_str());
        format!("{name} ({:?})", i.backend)
    }

    /// Error recorded outside an error scope (uncaptured error or lost device), if any.
    pub(crate) fn take_failure(&self) -> Option<String> {
        self.failure.lock().unwrap().take()
    }
}
