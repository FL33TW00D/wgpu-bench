use super::GPUHandle;
use std::collections::HashMap;
use tabled::settings::{object::Rows, Alignment, Modify, Panel, Style};
use tabled::{Table, Tabled};
use wgpu::QuerySet;

pub struct Profiler {
    handle: GPUHandle,
    query_set: QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_index: u32,
    timestamp_period: f32,
    query_to_node: HashMap<(u32, u32), (usize, String)>,
}

impl Profiler {
    pub fn new(handle: GPUHandle, count: u32) -> Self {
        let query_set = handle.device().create_query_set(&wgpu::QuerySetDescriptor {
            count: count * 2,
            ty: wgpu::QueryType::Timestamp,
            label: Some("PerfTimestamps"),
        });
        let timestamp_period = handle.queue().get_timestamp_period();

        let buffer_size = (count as usize * 2 * std::mem::size_of::<u64>()) as u64;
        let resolve_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfTimestamps"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let destination_buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfTimestamps"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            handle,
            query_set,
            resolve_buffer,
            destination_buffer,
            query_index: 0,
            timestamp_period,
            query_to_node: HashMap::with_capacity(count as usize),
        }
    }

    pub fn create_timestamp_queries(
        &mut self,
        id: usize,
        name: &str,
    ) -> wgpu::ComputePassTimestampWrites {
        let beginning_index = self.query_index;
        self.query_index += 1;
        let end_index = self.query_index;
        self.query_index += 1;

        let timestamp_writes = wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(beginning_index),
            end_of_pass_write_index: Some(end_index),
        };

        self.query_to_node
            .insert((beginning_index, end_index), (id, name.to_string()));

        timestamp_writes
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..self.query_index,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn read_timestamps(&self) {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.handle.device().poll(wgpu::Maintain::Wait);
        let timestamp_view = self
            .destination_buffer
            .slice(
                ..(std::mem::size_of::<u64>() * self.query_index as usize) as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let _timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);

        todo!();
    }
}
