use rand::distributions::uniform::SampleUniform;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::prelude::SeedableRng;
use rand::rngs::SmallRng;

use crate::storage::{CPUStorage, GPUStorage};
use crate::DType;
use crate::DataType;
use crate::{Shape, Storage};

#[derive(Clone)]
pub struct Tensor<S: Storage> {
    dt: DType,
    shape: Shape,
    storage: S,
}

impl<S: Storage> Tensor<S> {
    pub fn new(dt: DType, shape: Shape, storage: S) -> Self {
        Self { dt, shape, storage }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }
}

pub type CPUTensor = Tensor<CPUStorage>;

impl CPUTensor {
    pub unsafe fn uninitialized(dt: DType, shape: Shape, alignment: usize) -> anyhow::Result<Self> {
        let bytes = shape.numel() * dt.size_of();
        let layout = std::alloc::Layout::from_size_align(bytes, alignment)?;
        let data = if bytes == 0 {
            std::ptr::null()
        } else {
            let ptr = std::alloc::alloc(layout);
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        let storage = CPUStorage::new(data, layout);
        Ok(Tensor::new(dt, shape, storage))
    }

    pub fn from_slice<T: DataType>(data: &[T], shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let mut tensor =
            unsafe { Tensor::uninitialized(T::dt(), shape, T::dt().size_of()).unwrap() };
        tensor.storage_mut().as_bytes_mut().copy_from_slice(bytes);
        tensor
    }

    pub fn rand<T: num_traits::Float + DataType + SampleUniform>(shape: Shape) -> Self {
        let between = Uniform::from(T::from(-10).unwrap()..T::from(10).unwrap());
        let mut rng: SmallRng = SeedableRng::seed_from_u64(42);
        let rand_vec = (0..shape.numel())
            .map(|_| T::from(between.sample(&mut rng) / T::from(50).unwrap()).unwrap())
            .collect::<Vec<_>>();
        Self::from_slice(&rand_vec, shape)
    }
}

pub type GPUTensor = Tensor<GPUStorage>;
