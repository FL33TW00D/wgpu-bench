use std::{borrow::Cow, time::Duration};

use criterion::{BenchmarkId, Criterion};

use crate::{GPUHandle, GPUTensor, OpMetadata, WgpuTimer, Workload};

pub trait KernelContextExt {
    fn insert_workload(&mut self, workload: &Workload);
}

impl KernelContextExt for tera::Context {
    fn insert_workload(&mut self, workload: &Workload) {
        self.insert("workgroup_size_x", &workload.size().0);
        self.insert("workgroup_size_y", &workload.size().1);
        self.insert("workgroup_size_z", &workload.size().2);
    }
}

/// A trait for kernels that can be benchmarked
///
/// Each trait needs associated metadata that will be passed to the kernel.
/// The `tensors` method is used to define the input and outputs of the kernel.
pub trait Kernel: std::fmt::Debug {
    type Metadata: OpMetadata;
    fn name() -> &'static str;
    fn source(workload: &Workload) -> String;
    fn tensors(handle: &GPUHandle) -> Vec<GPUTensor>;
    fn workload(tensors: &[GPUTensor]) -> Workload;
    fn metadata(&self, tensors: &[GPUTensor]) -> Self::Metadata;
}

pub fn benchmark<K: Kernel>(c: &mut Criterion<&WgpuTimer>, timer: &WgpuTimer, kernel: K) {
    let handle = timer.handle();

    let tensors = K::tensors(handle);
    let workload = K::workload(&tensors);

    let mut buffers = tensors
        .iter()
        .map(|t| t.storage().inner())
        .collect::<Vec<_>>();

    let uniform_buffer = kernel.metadata(&tensors).into_buffer(handle);
    buffers.push(&uniform_buffer);

    let bind_group_entries = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: (i % 4) as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();

    let shader_module = unsafe {
        handle
            .device()
            .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&K::source(&workload))),
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

    let mut group = c.benchmark_group("wgpu kernel");
    group.warm_up_time(Duration::from_secs(2)); //Limit warmup time to avoid MAX_QUERIES limit
    group.bench_function(BenchmarkId::new(K::name(), 0), |b| {
        b.iter(|| {
            //We aren't worried about the overhead in here,
            //we only track the actual kernel execution time
            let mut encoder = handle
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(timer.timestamp_writes()),
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
            timer.increment_query();
        });
    });
    group.finish()
}
