use std::{alloc::Layout, ops::RangeBounds, sync::Arc};
use wgpu::{util::DeviceExt, Buffer, BufferAddress, BufferSlice, BufferUsages};

use crate::GPUHandle;

// Caution: no pooling of buffers is done for benchmarking
// long running benchmarks could OOM
pub trait Storage: std::fmt::Debug + Clone + 'static {
    fn to_gpu(self, handle: &GPUHandle) -> GPUStorage;
    fn to_cpu(self) -> CPUStorage;
    fn n_bytes(&self) -> usize;
}

#[derive(derive_new::new, Debug, PartialEq, Eq)]
pub struct CPUStorage(*mut u8, Layout);

impl CPUStorage {
    pub fn inner(&self) -> (*mut u8, Layout) {
        (self.0, self.1)
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1.size()) }
    }
}

impl Clone for CPUStorage {
    fn clone(&self) -> Self {
        let (ptr, layout) = self.inner();
        let alloc = unsafe { std::alloc::alloc(layout) };
        unsafe { ptr.copy_to_nonoverlapping(alloc, layout.size()) };

        Self(alloc, layout)
    }
}

impl Drop for CPUStorage {
    fn drop(&mut self) {
        if !self.0.is_null() && self.1.size() > 0 {
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
}

impl Storage for CPUStorage {
    //No allocations are pooled here because we don't care
    fn to_gpu(self, handle: &GPUHandle) -> GPUStorage {
        let mut min_bytes = [0; 16];
        let bytes = if self.as_bytes().len() < 16 {
            min_bytes[..self.as_bytes().len()].copy_from_slice(self.as_bytes());
            &min_bytes //&[u8]
        } else {
            self.as_bytes() //&[u8]
        };

        let buffer = handle
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });
        //These should be batched up
        handle.queue().submit(None);
        handle.device().poll(wgpu::Maintain::Wait);
        GPUStorage(buffer.into())
    }

    fn to_cpu(self) -> CPUStorage {
        self
    }

    fn n_bytes(&self) -> usize {
        self.1.size()
    }
}

#[derive(Debug, Clone)]
pub struct GPUBuffer(Arc<wgpu::Buffer>);

impl std::ops::Deref for GPUBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<wgpu::Buffer> for GPUBuffer {
    fn from(b: wgpu::Buffer) -> Self {
        Self(Arc::new(b))
    }
}

#[derive(Clone)]
pub struct GPUStorage(GPUBuffer);

impl From<GPUBuffer> for GPUStorage {
    fn from(b: GPUBuffer) -> Self {
        Self(b)
    }
}

impl std::fmt::Debug for GPUStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUStorage")
            .field("buffer", &self.0.global_id())
            .field("size", &self.0.size())
            .field("usage", &self.0.usage())
            .finish()
    }
}

impl PartialEq for GPUStorage {
    fn eq(&self, other: &Self) -> bool {
        self.0.global_id() == other.0.global_id()
    }
}

impl GPUStorage {
    pub fn new(buffer: GPUBuffer) -> Self {
        Self(buffer)
    }

    pub fn inner(&self) -> &GPUBuffer {
        &self.0
    }

    pub fn set_inner(&mut self, b: GPUBuffer) {
        self.0 = b;
    }

    pub fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.0.as_entire_binding()
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.0.usage()
    }

    pub fn slice<S: RangeBounds<wgpu::BufferAddress>>(&self, bounds: S) -> BufferSlice {
        self.0.slice(bounds)
    }

    pub fn unmap(&self) {
        self.0.unmap();
    }

    pub fn buffer_id(&self) -> wgpu::Id<Buffer> {
        self.0.global_id()
    }

    pub fn size(&self) -> BufferAddress {
        self.0.size()
    }
}

impl Storage for GPUStorage {
    fn to_gpu(self, _h: &GPUHandle) -> GPUStorage {
        self
    }

    fn to_cpu(self) -> CPUStorage {
        todo!()
    }

    fn n_bytes(&self) -> usize {
        self.0.size() as usize
    }
}
