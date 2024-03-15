use std::borrow::Cow;

use criterion::{BenchmarkId, Criterion, Throughput};

use crate::{CPUTensor, GPUBuffer, GPUHandle, GPUTensor, OpMetadata, WgpuTimer, Workload};

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

pub trait KernelBench: std::fmt::Debug {
    type Metadata: OpMetadata;
    fn name() -> &'static str;
    fn source(&self, workload: &Workload) -> String;
    fn tensors(&self) -> Vec<CPUTensor>;
    fn workload(&self, tensors: &[CPUTensor]) -> Workload;
    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata;
    fn validate(&self, tensors: &[CPUTensor]);
}

pub fn dispatch_validate<K: KernelBench>(handle: &GPUHandle, kernel: &K) -> Vec<GPUTensor> {
    let _ = env_logger::builder().is_test(true).try_init();
    let tensors = kernel.tensors();
    let workload = kernel.workload(&tensors);
    log::debug!("Workload: {:?}", workload);
    let source = kernel.source(&workload);
    log::debug!("Source: {}", source);
    let pipeline = source_to_pipeline(handle, &source);
    let uniform_buffer = kernel.metadata(&tensors).into_buffer(handle);
    let gpu_tensors = tensors
        .into_iter()
        .map(|t| t.into_gpu(handle))
        .collect::<Vec<_>>();
    let bind_groups = tensors_to_bind_groups(handle, &gpu_tensors, uniform_buffer, &pipeline);
    dispatch(handle, &workload, &bind_groups, &pipeline, None);
    gpu_tensors
}

#[inline(always)]
pub fn dispatch(
    handle: &GPUHandle,
    workload: &Workload,
    bind_groups: &[wgpu::BindGroup],
    pipeline: &wgpu::ComputePipeline,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
) {
    let mut encoder = handle
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes,
        });
        for (i, bind_group) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(i as _, bind_group, &[]);
        }
        cpass.set_pipeline(pipeline);
        let (x, y, z) = workload.count().as_tuple();
        for _ in 0..WgpuTimer::COMPUTE_PER_QUERY {
            cpass.dispatch_workgroups(x, y, z);
        }
    }
    handle.queue().submit(Some(encoder.finish()));
    handle.device().poll(wgpu::Maintain::Wait);
}

pub fn source_to_pipeline(handle: &GPUHandle, source: &str) -> wgpu::ComputePipeline {
    let shader_module = unsafe {
        handle
            .device()
            .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            })
    };

    handle
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        })
}

pub fn tensors_to_bind_groups(
    handle: &GPUHandle,
    tensors: &[GPUTensor],
    uniform_buffer: GPUBuffer,
    pipeline: &wgpu::ComputePipeline,
) -> Vec<wgpu::BindGroup> {
    let mut bind_group_entries = vec![];
    for tensor in tensors {
        bind_group_entries.append(&mut tensor.bindings());
    }

    let mut standard_bind_groups = bind_group_entries
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

    let uniform_bind_group = handle
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(standard_bind_groups.len() as _),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
    standard_bind_groups.push(uniform_bind_group);
    standard_bind_groups
}

pub fn benchmark<K: KernelBench>(
    c: &mut Criterion<&WgpuTimer>,
    timer: &WgpuTimer,
    kernel: K,
    throughput: Throughput,
) {
    let handle = timer.handle();
    let tensors = kernel.tensors();
    kernel.validate(&tensors);
    let workload = kernel.workload(&tensors);
    let source = kernel.source(&workload);
    let pipeline = source_to_pipeline(handle, &source);
    let uniform_buffer = kernel.metadata(&tensors).into_buffer(handle);

    let gpu_tensors = tensors
        .into_iter()
        .map(|t| t.into_gpu(handle))
        .collect::<Vec<_>>();
    let bind_groups = tensors_to_bind_groups(handle, &gpu_tensors, uniform_buffer, &pipeline);

    let mut group = c.benchmark_group(K::name());
    group.throughput(throughput);
    group.bench_function(BenchmarkId::new(K::name(), 0), |b| {
        b.iter(|| {
            let tsw = timer.timestamp_writes();
            dispatch(handle, &workload, &bind_groups, &pipeline, Some(tsw));
            timer.increment_query();
        });
    });
    group.finish()
}
