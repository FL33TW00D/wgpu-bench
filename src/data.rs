use num_traits::Float;
use rand::distributions::{uniform::SampleUniform, Distribution, Standard, Uniform};
use wgpu::util::DeviceExt;

use crate::GPUHandle;

pub fn generate_weight_data<F: Float + bytemuck::Pod + std::fmt::Debug>(elements: usize) -> Vec<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(F::from(-10.0).unwrap()..F::from(10.0).unwrap());
    let x: Vec<F> = (0..elements).map(|_| dist.sample(&mut rng)).collect();
    x
}

pub fn empty_buffer<F: Float + bytemuck::Pod + std::fmt::Debug>(
    device: &wgpu::Device,
    elements: usize,
) -> wgpu::Buffer
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let data = vec![F::zero(); elements];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

pub fn rand_gpu_buffer<F: Float + bytemuck::Pod + std::fmt::Debug>(
    handle: &GPUHandle,
    elements: usize,
) -> wgpu::Buffer
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
