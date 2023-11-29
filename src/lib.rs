mod data;
mod handle;
mod profiler;
mod shape;

use std::cell::Cell;

pub use data::*;
pub use handle::*;
pub use profiler::*;
pub use shape::*;

use criterion::{
    measurement::{Measurement, ValueFormatter},
    Throughput,
};
use wgpu::QuerySet;

/// Start and end index in the counter sample buffer
pub struct TimerPair {
    pub start: u32,
    pub end: u32,
}

pub struct WgpuTimer {
    handle: GPUHandle,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_index: Cell<u32>,
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
        let timestamp_view = self
            .destination_buffer
            .slice(
                i.start as u64..(std::mem::size_of::<u64>() as u32 * i.end) as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
        let [start, end] = timestamps.try_into().unwrap();
        end - start // is this right?
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
        todo!()
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
