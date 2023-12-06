use std::sync::Arc;

use wgpu::Adapter;
use wgpu::DeviceType;

use wgpu::Limits;

/// # GPUHandle
///
/// A reference counted handle to a GPU device and queue.
#[derive(Debug, Clone)]
pub struct GPUHandle {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl std::ops::Deref for GPUHandle {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl GPUHandle {
    fn get_features() -> wgpu::Features {
        wgpu::Features::default()
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::SUBGROUP_COMPUTE
    }

    pub async fn new() -> Result<Self, anyhow::Error> {
        let adapter = Self::select_adapter();

        let mut device_descriptor = wgpu::DeviceDescriptor {
            label: Some("rumble"),
            required_features: Self::get_features(),
            required_limits: Limits {
                max_buffer_size: (2 << 29) - 1,
                max_storage_buffer_binding_size: (2 << 29) - 1,
                ..Default::default()
            },
        };
        let device_request = adapter.request_device(&device_descriptor, None).await;
        let (device, queue) = if let Err(e) = device_request {
            log::warn!("Failed to create device with error: {:?}", e);
            log::warn!("Trying again with reduced limits");
            device_descriptor.required_limits = adapter.limits();
            let device_request = adapter.request_device(&device_descriptor, None).await;
            device_request.unwrap()
        } else {
            device_request.unwrap()
        };

        Ok(Self {
            inner: Arc::new(Inner { device, queue }),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn select_adapter() -> Adapter {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            ..Default::default()
        });
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);

        let adapter = {
            let mut most_performant_adapter = None;
            let mut current_score = -1;

            instance.enumerate_adapters(backends).for_each(|adapter| {
                let info = adapter.get_info();
                let score = match info.device_type {
                    DeviceType::DiscreteGpu => 5,
                    DeviceType::Other => 4, //Other is usually discrete
                    DeviceType::IntegratedGpu => 3,
                    DeviceType::VirtualGpu => 2,
                    DeviceType::Cpu => 1,
                };

                if score > current_score {
                    most_performant_adapter = Some(adapter);
                    current_score = score;
                }
            });

            if let Some(adapter) = most_performant_adapter {
                adapter
            } else {
                panic!("No adapter found, please check if your GPU is supported");
            }
        };
        log::info!("Using adapter {:?}", adapter.get_info());
        adapter
    }
}
