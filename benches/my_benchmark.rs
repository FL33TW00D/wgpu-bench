#![feature(const_async_blocks)]
use std::borrow::Cow;

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

    let n_elements = 1024 * 1024;
    let A = rand_gpu_buffer::<f32>(TIMER.handle(), n_elements);
    let B = rand_gpu_buffer::<f32>(TIMER.handle(), 4);
    let C = empty_buffer::<f32>(TIMER.handle().device(), n_elements);

    let shader_module = unsafe {
        TIMER
            .handle()
            .device()
            .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../add.wgsl"))),
            })
    };

    let pipeline =
        TIMER
            .handle()
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });

    let bind_group_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: A.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: B.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: C.as_entire_binding(),
        },
    ];

    let bind_group = TIMER
        .handle()
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bind_group_entries,
        });

    group.bench_function(BenchmarkId::new("add kernel", n_elements), |b| {
        b.iter(|| {
            let mut encoder = TIMER
                .handle()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let tsw = TIMER.timestamp_writes();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(tsw),
                });
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.set_pipeline(&pipeline);
                cpass.dispatch_workgroups((n_elements / 64) as u32, 1, 1);
            }
            TIMER.increment_query();
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
