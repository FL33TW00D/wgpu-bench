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
    fn source(workload: &Workload) -> String;
    fn tensors(&self) -> Vec<CPUTensor>;
    fn workload(tensors: &[CPUTensor]) -> Workload;
    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata;
    fn validate(&self, tensors: &[CPUTensor]);
}

pub fn dispatch_validate<K: KernelBench>(handle: &GPUHandle, kernel: &K) -> Vec<GPUTensor> {
    let tensors = kernel.tensors();
    let workload = K::workload(&tensors);
    let source = K::source(&workload);
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
    let mut buffers = tensors
        .iter()
        .map(|t| t.storage().inner())
        .collect::<Vec<_>>();
    buffers.push(&uniform_buffer);

    let bind_group_entries = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: (i % 4) as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();

    bind_group_entries
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
        .collect::<Vec<_>>()
}

pub fn benchmark<K: KernelBench>(
    c: &mut Criterion<&WgpuTimer>,
    timer: &WgpuTimer,
    kernel: K,
    bytes_per_iter: usize,
) {
    let handle = timer.handle();
    let tensors = kernel.tensors();
    kernel.validate(&tensors);
    let workload = K::workload(&tensors);
    let source = K::source(&workload);
    let pipeline = source_to_pipeline(handle, &source);
    let uniform_buffer = kernel.metadata(&tensors).into_buffer(handle);

    let gpu_tensors = tensors
        .into_iter()
        .map(|t| t.into_gpu(handle))
        .collect::<Vec<_>>();
    let bind_groups = tensors_to_bind_groups(handle, &gpu_tensors, uniform_buffer, &pipeline);

    let mut group = c.benchmark_group("wgpu kernel");
    group.throughput(Throughput::Bytes(bytes_per_iter as u64));
    //group.warm_up_time(Duration::from_secs(2)); //Limit warmup time to avoid MAX_QUERIES limit
    group.bench_function(BenchmarkId::new(K::name(), 0), |b| {
        b.iter(|| {
            let tsw = timer.timestamp_writes();
            dispatch(handle, &workload, &bind_groups, &pipeline, Some(tsw));
            timer.increment_query();
        });
    });
    group.finish()
}
