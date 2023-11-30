#![allow(non_snake_case)]
use encase::ShaderType;

use criterion::{criterion_group, criterion_main, Criterion};
use wgpu_bencher::{empty_buffer, rand_gpu_buffer, GPUHandle, Kernel, OpMetadata, WgpuTimer};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType)]
pub struct LayerNormMeta {
    N: u32,
}

impl OpMetadata for LayerNormMeta {}

pub struct LayerNorm;

impl Kernel for LayerNorm {
    type Metadata = LayerNormMeta;

    fn name() -> &'static str {
        "LayerNorm"
    }

    fn source() -> &'static str {
        include_str!("../add.wgsl")
    }

    fn buffers(handle: &GPUHandle) -> Vec<wgpu::Buffer> {
        let A = rand_gpu_buffer::<f32>(handle, 1024);
        let B = rand_gpu_buffer::<f32>(handle, 4);
        let C = empty_buffer::<f32>(handle.device(), 1024);
        vec![A, B, C]
    }
}

pub fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let layernorm = LayerNorm;
    wgpu_bencher::benchmark(c, TIMER.handle(), layernorm)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
