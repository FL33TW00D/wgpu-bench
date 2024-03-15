use encase::{private::WriteInto, ShaderType, UniformBuffer};

use crate::{GPUBuffer, GPUHandle};

pub const UNIFORM_ALIGN: usize = 256;
pub const STORAGE_BUFFER_ALIGN: usize = 256;
pub const MIN_STORAGE_BUFFER_SIZE: usize = 16;

pub trait OpMetadata: Sized + ShaderType + WriteInto + std::fmt::Debug {
    fn into_buffer(&self, handle: &GPUHandle) -> GPUBuffer {
        let size: usize = self.size().get() as _;
        let aligned_size = size + (UNIFORM_ALIGN - size % UNIFORM_ALIGN);

        let mut uniform = UniformBuffer::new(Vec::with_capacity(aligned_size));
        uniform.write(self).unwrap();

        let buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        handle
            .queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&uniform.into_inner()));
        buffer.into()
    }
}
