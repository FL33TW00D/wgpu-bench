use num_traits::Float;
use rand::distributions::{uniform::SampleUniform, Distribution, Standard, Uniform};
use wgpu::util::DeviceExt;

use crate::GPUHandle;

pub fn generate_weight_data<F: Float + bytemuck::Pod>(elements: usize) -> Vec<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(F::from(-10.0).unwrap()..F::from(10.0).unwrap());
    (0..elements).map(|_| dist.sample(&mut rng)).collect()
}

fn rand_gpu_buffer<F: Float + bytemuck::Pod>(handle: &GPUHandle, elements: usize) -> wgpu::Buffer
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let data = generate_weight_data::<F>(elements);
    let buffer = handle
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    handle.queue().submit(None);
    handle.device().poll(wgpu::Maintain::Wait);
    buffer
}
