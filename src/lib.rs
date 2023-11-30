mod data;
mod handle;
mod metadata;
mod profiler;
mod shape;

use std::{cell::Cell, ops::Range};

pub use data::*;
pub use handle::*;
pub use metadata::*;
pub use profiler::*;
pub use shape::*;

use criterion::{
    measurement::{Measurement, ValueFormatter},
    Throughput,
};
use wgpu::QuerySet;

pub const MAX_QUERIES: u32 = 512;

/// Start and end index in the counter sample buffer
#[derive(Debug, Clone, Copy)]
pub struct QueryPair {
    pub start: u32,
    pub end: u32,
}

impl QueryPair {
    pub fn first() -> Self {
        Self { start: 0, end: 1 }
    }

    pub fn start_addr(&self) -> wgpu::BufferAddress {
        self.start as u64 * std::mem::size_of::<u64>() as wgpu::BufferAddress
    }

    pub fn end_addr(&self) -> wgpu::BufferAddress {
        self.end as u64 * std::mem::size_of::<u64>() as wgpu::BufferAddress
    }

    pub fn size(&self) -> wgpu::BufferAddress {
        ((self.end - self.start) as usize * std::mem::size_of::<u64>()) as wgpu::BufferAddress
    }
}

impl Into<Range<u32>> for QueryPair {
    fn into(self) -> Range<u32> {
        self.start..self.end
    }
}

pub struct WgpuTimer {
    handle: GPUHandle,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    current_query: Cell<QueryPair>,
}

unsafe impl Send for WgpuTimer {}
unsafe impl Sync for WgpuTimer {}

impl WgpuTimer {
    pub fn new(handle: GPUHandle) -> Self {
        let query_set = handle.device().create_query_set(&wgpu::QuerySetDescriptor {
            count: MAX_QUERIES,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        });

        let size = (MAX_QUERIES as usize * 2 * std::mem::size_of::<u64>()) as wgpu::BufferAddress;

        let resolve_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let destination_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            handle,
            query_set,
            resolve_buffer,
            destination_buffer,
            current_query: QueryPair::first().into(),
        }
    }

    pub fn resolve_pair(&self, encoder: &mut wgpu::CommandEncoder, pair: QueryPair) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn handle(&self) -> &GPUHandle {
        &self.handle
    }

    pub fn query_set(&self) -> &QuerySet {
        &self.query_set
    }

    pub fn timestamp_writes(&self) -> wgpu::ComputePassTimestampWrites {
        let pair = self.current_query.get();
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(pair.start),
            end_of_pass_write_index: Some(pair.end),
        }
    }

    pub fn increment_query(&self) {
        let pair = self.current_query.get();
        self.current_query.set(QueryPair {
            start: pair.start + 2,
            end: pair.end + 2,
        });
    }

    pub fn current_query(&self) -> QueryPair {
        self.current_query.get()
    }
}

impl Measurement for &WgpuTimer {
    type Intermediate = QueryPair;

    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        let start = self.current_query.get();
        start
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.handle.device().poll(wgpu::Maintain::Wait);
        let timestamps: Vec<u64> = {
            println!("Mapping: {:?}", i.start_addr()..i.end_addr());
            let timestamp_view = self
                .destination_buffer
                .slice(i.start_addr()..i.end_addr())
                .get_mapped_range();

            (*bytemuck::cast_slice(&timestamp_view)).to_vec()
        };
        self.destination_buffer.unmap();
        self.increment_query();
        println!("Timestamps: {:?}", timestamps);
        1
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        (self.handle.queue().get_timestamp_period() as f64) * (*value as f64)
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &WgpuTimerFormatter
    }
}

struct WgpuTimerFormatter;

impl ValueFormatter for WgpuTimerFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.4} ms", value)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        match throughput {
            Throughput::Bytes(b) => format!(
                "{:.4} GiB/s",
                (*b as f64) / (1024.0 * 1024.0 * 1024.0) / (value * 1e-3)
            ),
            Throughput::Elements(b) => format!("{:.4} elements/s", (*b as f64) / (value * 1e-3)),
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "ms"
    }

    /// TODO!
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Bytes(_) => "GiB/s",
            Throughput::Elements(_) => "elements/s",
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "ms"
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::*;

    #[test]
    pub fn does_it_work() {
        let TIMER = WgpuTimer::new(pollster::block_on(async {
            GPUHandle::new().await.unwrap()
        }));

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

        let mut encoder = TIMER
            .handle()
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let tsw = TIMER.timestamp_writes();
            println!("Timestamp writes: {:?}", tsw);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(tsw),
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&pipeline);
            cpass.dispatch_workgroups((n_elements / 64) as u32, 1, 1);
        }
        TIMER.resolve_pair(&mut encoder, TIMER.current_query());
        TIMER.handle().queue().submit(Some(encoder.finish()));
        TIMER.handle().device().poll(wgpu::Maintain::Wait);

        TIMER
            .destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        TIMER.handle.device().poll(wgpu::Maintain::Wait);
        let timestamps: Vec<u64> = {
            let timestamp_view = TIMER.destination_buffer.slice(0..8).get_mapped_range();

            (*bytemuck::cast_slice(&timestamp_view)).to_vec()
        };
        TIMER.destination_buffer.unmap();
        TIMER.increment_query();
        println!("Timestamps: {:?}", timestamps);
    }
}
