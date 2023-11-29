#![feature(const_async_blocks)]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wgpu_bencher::{empty_buffer, rand_gpu_buffer, GPUHandle, WgpuTimer};

use criterion::BenchmarkId;

lazy_static::lazy_static! {
    pub static ref GPU_HANDLE: GPUHandle = pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    });
}

/// Profiles a simple element-wise addition kernel
pub fn cuda_bench(c: &mut Criterion<WgpuTimer>) {
    let mut group = c.benchmark_group("wgpu kernel");

    let buffer_size = 1024;
    let x = rand_gpu_buffer::<f32>(&GPU_HANDLE, buffer_size);
    let y = rand_gpu_buffer::<f32>(&GPU_HANDLE, buffer_size);
    let result = empty_buffer::<f32>(GPU_HANDLE.device(), buffer_size);

    group.throughput(Throughput::Bytes(buffer_size as u64 * 4));
    group.bench_function(BenchmarkId::new("add kernel", buffer_size), |b| {
        b.iter(|| {});
    });

    group.finish()
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(WgpuTimer::new(GPU_HANDLE.clone()));
    targets = cuda_bench
);
criterion_main!(bench);
