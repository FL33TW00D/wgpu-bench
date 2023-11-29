mod data;
mod handle;
mod profiler;
mod shape;

use std::{cell::Cell, ops::Range};

pub use data::*;
pub use handle::*;
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
pub struct TimerPair {
    pub start: u32,
    pub end: u32,
}

impl TimerPair {
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

impl Into<Range<u32>> for TimerPair {
    fn into(self) -> Range<u32> {
        self.start..self.end
    }
}

pub struct WgpuTimer {
    handle: GPUHandle,
    query_set: QuerySet,
    query_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_index: Cell<u32>,
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

        let query_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
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
            query_buffer,
            destination_buffer,
            query_index: 0.into(),
        }
    }

    pub fn resolve_pair(&self, encoder: &mut wgpu::CommandEncoder, pair: TimerPair) {
        encoder.resolve_query_set(&self.query_set, pair.into(), &self.query_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.query_buffer,
            pair.start_addr(),
            &self.destination_buffer,
            pair.end_addr(),
            pair.size(),
        );
    }

    pub fn handle(&self) -> &GPUHandle {
        &self.handle
    }
}

impl Measurement for WgpuTimer {
    type Intermediate = TimerPair;

    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        let index = self.query_index.get();
        let beginning_of_pass_write_index = index;
        let end_of_pass_write_index = index + 1;
        self.query_index.set(index + 2);
        TimerPair {
            start: beginning_of_pass_write_index,
            end: end_of_pass_write_index,
        }
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.handle.device().poll(wgpu::Maintain::Wait);
        let timestamps = {
            println!("Mapping: {:?}", i);
            let timestamp_view = self
                .destination_buffer
                .slice(i.start_addr()..i.end_addr())
                .get_mapped_range();

            (*bytemuck::cast_slice(&timestamp_view)).to_vec()
        };
        self.destination_buffer.unmap();
        timestamps[0]
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
