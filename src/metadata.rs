use encase::{private::WriteInto, ShaderType, UniformBuffer};

use crate::GPUHandle;

pub const UNIFORM_ALIGN: usize = 256;

pub trait OpMetadata: Sized + ShaderType + WriteInto + std::fmt::Debug {
    const __IS_VALID_META: () = {
        assert!(std::mem::size_of::<Self>() <= UNIFORM_ALIGN);
    };

    fn n_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn into_buffer(&self, handle: &GPUHandle) -> wgpu::Buffer {
        let mut cpu_uniform = UniformBuffer::new(Vec::with_capacity(self.n_bytes()));

        cpu_uniform.write(self).unwrap();

        let buffer = handle.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.n_bytes() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        handle
            .queue()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&cpu_uniform.into_inner()));
        buffer
    }
}
