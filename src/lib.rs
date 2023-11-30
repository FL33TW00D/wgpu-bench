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

pub const MAX_QUERIES: u32 = 4096;

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

    pub fn size(&self) -> wgpu::BufferAddress {
        ((self.end - self.start + 1) as usize * std::mem::size_of::<u64>()) as wgpu::BufferAddress
    }
}

impl Into<Range<u32>> for QueryPair {
    fn into(self) -> Range<u32> {
        self.start..self.end + 1
    }
}

pub struct WgpuTimer {
    handle: GPUHandle,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    current_query: Cell<QueryPair>,
}

//TODO: dumb
unsafe impl Send for WgpuTimer {}
unsafe impl Sync for WgpuTimer {}

impl WgpuTimer {
    pub fn new(handle: GPUHandle) -> Self {
        let query_set = handle.device().create_query_set(&wgpu::QuerySetDescriptor {
            count: MAX_QUERIES,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        });

        let resolve_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let destination_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
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

    pub fn resolve_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let resolve_range = self.current_query().into();
        println!("\nResolving query range: {:?}", resolve_range);
        encoder.resolve_query_set(&self.query_set, resolve_range, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.current_query().size(),
        );
    }

    pub fn handle(&self) -> &GPUHandle {
        &self.handle
    }

    pub fn query_set(&self) -> &QuerySet {
        &self.query_set
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
    type Intermediate = (); //query index

    type Value = u64; // Raw unscaled GPU counter
                      // Must be multiplied by the timestamp period to get nanoseconds

    fn start(&self) -> Self::Intermediate {
        let mut encoder = self
            .handle
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.write_timestamp(self.query_set(), self.current_query().start);
        self.handle().queue().submit(Some(encoder.finish()));
        self.handle.device().poll(wgpu::Maintain::Wait);
    }

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        let mut encoder = self
            .handle
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.write_timestamp(self.query_set(), self.current_query().end);
        self.resolve_pass(&mut encoder);
        self.handle().queue().submit(Some(encoder.finish()));
        self.handle.device().poll(wgpu::Maintain::Wait);

        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.handle.device().poll(wgpu::Maintain::Wait);
        let timestamps: Vec<u64> = {
            let timestamp_view = self.destination_buffer.slice(0..16).get_mapped_range();
            (*bytemuck::cast_slice(&timestamp_view)).to_vec()
        };
        self.destination_buffer.unmap();
        self.increment_query();
        println!("Timestamps: {:?}", timestamps);
        let delta = timestamps[1] - timestamps[0];
        delta
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
        format!("{:.4} ns", value)
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
        "ns"
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
        "ns"
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    pub fn pair_size() {
        let query = QueryPair::first();
        assert_eq!(query.size(), 16);
    }
}
