#![feature(const_async_blocks)]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wgpu_bencher::{empty_buffer, rand_gpu_buffer, GPUHandle, WgpuTimer};

use criterion::BenchmarkId;

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

/// Profiles a simple element-wise addition kernel
pub fn wgpu_bench(c: &mut Criterion<&WgpuTimer>) {
    let mut group = c.benchmark_group("wgpu kernel");

    let buffer_size = 1024;
    let x = rand_gpu_buffer::<f32>(TIMER.handle(), buffer_size);
    let y = rand_gpu_buffer::<f32>(TIMER.handle(), buffer_size);
    let result = empty_buffer::<f32>(TIMER.handle().device(), buffer_size);

    group.throughput(Throughput::Bytes(buffer_size as u64 * 4));
    group.bench_function(BenchmarkId::new("add kernel", buffer_size), |b| {
        b.iter(|| {
            let mut encoder = TIMER
                .handle()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(TIMER.timestamp_writes()),
                });
                /*
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.set_pipeline(pipeline);
                cpass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
                */
            }

            TIMER.handle().queue().submit(Some(encoder.finish()));
            TIMER.handle().device().poll(wgpu::Maintain::Wait);
        });
    });

    group.finish()
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = wgpu_bench
);
criterion_main!(bench);
