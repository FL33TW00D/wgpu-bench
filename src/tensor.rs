use numpy::ndarray::{ArrayD, ArrayViewD};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    prelude::{Distribution, SeedableRng},
    rngs::SmallRng,
};

use numpy::PyArrayDyn;

use crate::storage::{CPUStorage, GPUStorage};
use crate::DType;
use crate::DataType;
use crate::GPUHandle;
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

    pub fn dt(&self) -> DType {
        self.dt
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

    pub fn n_bytes(&self) -> usize {
        self.shape().numel() * self.dt().size_of()
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

    pub fn zeros<D: DataType>(shape: Shape) -> Self {
        let data = vec![D::zero(); shape.numel()];
        Self::from_slice(&data, shape)
    }

    pub fn into_gpu(self, handle: &GPUHandle) -> GPUTensor {
        let storage = self.storage.to_gpu(handle);
        GPUTensor::new(self.dt, self.shape.clone(), storage)
    }

    pub unsafe fn into_array_unchecked<D: DataType>(self) -> ArrayD<D> {
        self.to_array_view_unchecked::<D>().to_owned()
    }

    pub unsafe fn to_array_view_unchecked<T: DataType>(&self) -> ArrayViewD<T> {
        let inner = self.storage().inner();
        if self.n_bytes() != 0 {
            ArrayViewD::from_shape_ptr(self.shape().to_vec(), inner.0 as *const T)
        } else {
            ArrayViewD::from_shape(self.shape().to_vec(), &[]).unwrap()
        }
    }

    pub fn to_py<'s, 'p: 's, T: DataType + numpy::Element>(
        &'s self,
        py: &'p pyo3::Python<'p>,
    ) -> &PyArrayDyn<T> {
        use numpy::PyArray;
        PyArray::from_owned_array(*py, unsafe { self.clone().into_array_unchecked::<T>() })
    }
}

impl<T: DataType + numpy::Element> From<&PyArrayDyn<T>> for CPUTensor {
    fn from(array: &PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

impl<T: DataType + numpy::Element> From<PyArrayDyn<T>> for CPUTensor {
    fn from(array: PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

impl<T: DataType> From<ArrayD<T>> for CPUTensor {
    fn from(it: ArrayD<T>) -> Self {
        if it.as_slice().is_some() {
            let layout = std::alloc::Layout::from_size_align(
                it.len() * std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let shape = it.shape().into();
            let vec = it.into_raw_vec().into_boxed_slice();
            let data = Box::into_raw(vec) as *mut u8;

            Tensor::new(T::dt(), shape, CPUStorage::new(data, layout))
        } else {
            panic!("Cannot convert numpy array with non-contiguous memory layout to tensor");
        }
    }
}

pub type GPUTensor = Tensor<GPUStorage>;
