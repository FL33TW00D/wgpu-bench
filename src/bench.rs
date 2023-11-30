use std::borrow::Cow;

use criterion::{BenchmarkId, Criterion};

use crate::{GPUHandle, OpMetadata, WgpuTimer, WorkgroupCount, WorkgroupSize, Workload};

pub trait Kernel {
    type Metadata: OpMetadata;
    type Problem;
    fn name() -> &'static str;
    fn workload() -> Workload;
    fn source(workload: Workload) -> &'static str;
    fn problem() -> Self::Problem;
    fn metadata(&self) -> Self::Metadata;
    fn buffers(&self, handle: &GPUHandle) -> Vec<wgpu::Buffer>;
}

pub fn benchmark<K: Kernel>(c: &mut Criterion<&WgpuTimer>, handle: &GPUHandle, kernel: K) {
    let shader_module = unsafe {
        handle
            .device()
            .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(K::source(K::workload()))),
            })
    };

    let pipeline = handle
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

    let mut buffers = kernel.buffers(handle);
    let uniform_buffer = kernel.metadata().into_buffer(handle);
    buffers.push(uniform_buffer);

    let bind_group_entries = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: (i % 4) as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();

    let bind_groups = bind_group_entries
        .chunks(4)
        .enumerate()
        .map(|(i, entries)| {
            handle
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(i as _),
                    entries,
                })
        })
        .collect::<Vec<_>>();

    let workload = K::workload();

    let mut group = c.benchmark_group("wgpu kernel");
    group.bench_function(BenchmarkId::new(K::name(), 0), |b| {
        b.iter(|| {
            let mut encoder = handle
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                for (i, bind_group) in bind_groups.iter().enumerate() {
                    cpass.set_bind_group(i as _, bind_group, &[]);
                }
                cpass.set_pipeline(&pipeline);
                let (x, y, z) = workload.count().as_tuple();
                cpass.dispatch_workgroups(x, y, z);
            }
            handle.queue().submit(Some(encoder.finish()));
            handle.device().poll(wgpu::Maintain::Wait);
        });
    });
    group.finish()
}
