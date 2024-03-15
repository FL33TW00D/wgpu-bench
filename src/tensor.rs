use bytemuck::NoUninit;
use ndarray::Dimension;
use numpy::ndarray::{ArrayD, ArrayViewD};
use rand::{distributions::uniform::SampleUniform, prelude::SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Poisson};

use numpy::PyArrayDyn;
use wgpu::{BindGroupEntry, BindingResource, BufferUsages};

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

unsafe impl<S: Storage> Send for Tensor<S> {}
unsafe impl<S: Storage> Sync for Tensor<S> {}

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

    pub fn into_inner(self) -> (DType, Shape, S) {
        let Self { dt, shape, storage } = self;
        (dt, shape, storage)
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

    pub fn to_vec<T: DataType>(&self) -> anyhow::Result<Vec<T>> {
        let bytes = self.storage().as_bytes();
        let data = bytemuck::cast_slice(bytes);
        Ok(data.to_vec())
    }

    pub fn from_slice<T: DataType>(data: &[T], shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let mut tensor =
            unsafe { Tensor::uninitialized(T::dt(), shape, T::dt().size_of()).unwrap() };
        tensor.storage_mut().as_bytes_mut().copy_from_slice(bytes);
        tensor
    }

    pub unsafe fn from_quantized<T: DataType, U: AsRef<[T]>>(
        data: U,
        shape: Shape,
        dt: DType,
    ) -> CPUTensor {
        let raw_data = data.as_ref();
        let data_bytes: &[u8] = bytemuck::cast_slice(raw_data);
        let n_bytes = data_bytes.len();

        let layout = std::alloc::Layout::from_size_align(n_bytes, dt.size_of()).unwrap();
        let data = if n_bytes == 0 {
            std::ptr::null()
        } else {
            let ptr = std::alloc::alloc(layout);
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        let storage = CPUStorage::new(data, layout);
        let mut tensor = Tensor::new(dt, shape, storage);
        tensor
            .storage_mut()
            .as_bytes_mut()
            .copy_from_slice(data_bytes);
        tensor
    }

    pub fn randn<T: num_traits::Float + DataType + SampleUniform>(shape: Shape) -> Self {
        let between = Poisson::new(11.0).unwrap();
        let mut rng: SmallRng = SeedableRng::seed_from_u64(42);
        let rand_vec = (0..shape.numel())
            .map(|_| T::from(between.sample(&mut rng)).unwrap())
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

    pub fn fmt(&self) -> String {
        format!("{}", unsafe { self.to_array_view_unchecked::<f32>() })
    }

    pub fn debug_fmt(&self) -> String {
        format!("{:?}", unsafe { self.to_array_view_unchecked::<f32>() })
    }

    pub fn all_close(&self, other: &Self, atol: f32, rtol: f32) -> anyhow::Result<()> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
        }
        let ma = unsafe { self.to_array_view_unchecked::<f32>() };
        let mb = unsafe { other.to_array_view_unchecked::<f32>() };
        let mut elem_cnt = 0;
        let mut fail_cnt = 0;
        let mut total_error = 0f32;
        let mut mae = -1f32;
        let mut mae_idxs = Default::default();
        ndarray::indices_of(&ma).into_iter().try_for_each(|idxs| {
            let (a, b) = (ma[&idxs], mb[&idxs]);
            let abs_diff = (a - b).abs();
            let cur_mae = mae.max(abs_diff);
            if cur_mae > mae {
                mae = cur_mae;
                mae_idxs = idxs.clone();
            }
            total_error += abs_diff;
            elem_cnt += 1;

            if !((a.is_nan() && b.is_nan())
                || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
                || abs_diff <= atol + rtol * b.abs())
            {
                let slice = idxs.slice();
                log::trace!(
                    "Mismatch at {:?}: {:?} != {:?} (atol={}, rtol={})",
                    slice,
                    a,
                    b,
                    atol,
                    rtol
                );
                fail_cnt += 1;
            }
            Ok::<(), anyhow::Error>(())
        })?;
        let avg_error = total_error / elem_cnt as f32;
        let slice = mae_idxs.slice();
        if fail_cnt > 0 {
            anyhow::bail!(
                "{} samples not close - AVGE={} MAE={} at {:?}",
                fail_cnt,
                avg_error,
                mae,
                slice,
            );
        } else {
            println!("All close - AVGE={} MAE={} at {:?}", avg_error, mae, slice,);
            Ok(())
        }
    }
}

impl std::fmt::Debug for CPUTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CPUTensor")
            .field("dt", &self.dt)
            .field("shape", &self.shape)
            .field("storage", &self.storage)
            .finish()
    }
}

impl std::fmt::Display for CPUTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.fmt())
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

impl GPUTensor {
    /// # Bindings
    ///
    /// Only applicable to GPU tensors.
    /// Generates the bind group entries required to bind the tensor to a kernel.
    /// Quantized tensors may use multiple bind groups.
    /// Unquantized tensors should only use a single bind group.
    pub(crate) fn bindings(&self, current_binding: usize) -> Vec<BindGroupEntry> {
        let buf = self.storage().inner();
        let numel = self.shape().numel();
        let segments = self.dt().segments(numel, buf.size() as usize);

        let mut entries = vec![];
        for (idx, seg) in segments.iter().enumerate() {
            let (offset, size) = (seg.offset, seg.size);
            entries.push(BindGroupEntry {
                binding: ((current_binding + idx) % 4) as _,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset,
                    size,
                }),
            });
        }
        entries
    }

    fn read_to_host<A: NoUninit>(shape: Shape, dt: DType, bytes: &[A]) -> CPUTensor {
        match dt {
            DType::F32 => CPUTensor::from_slice::<f32>(bytemuck::cast_slice(bytes), shape),
            DType::I32 => CPUTensor::from_slice::<i32>(bytemuck::cast_slice(bytes), shape),
            DType::U32 => CPUTensor::from_slice::<u32>(bytemuck::cast_slice(bytes), shape),
            _ => panic!("Unsupported dtype"),
        }
    }

    fn into_cpu_inner(self, handle: &GPUHandle) -> anyhow::Result<CPUTensor> {
        let (dt, shape, storage) = self.into_inner();
        if !storage.usage().contains(BufferUsages::COPY_SRC) {
            panic!("Attempted to read GPU tensor to host without COPY_SRC usage")
        }
        let buffer_slice = storage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        wgpu::util::DownloadBuffer::read_buffer(
            handle.device(),
            handle.queue(),
            &buffer_slice,
            move |buffer| {
                // Called on download completed
                tx.send(match buffer {
                    Ok(db) => Ok(Self::read_to_host(shape, dt, &db)),
                    Err(error) => panic!("Failed to read GPU tensor to host: {:?}", error),
                })
                .unwrap();
            },
        );
        handle.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap()
    }

    ///Consumes the GPU tensor and returns a CPU tensor
    pub fn into_cpu(self, handle: &GPUHandle) -> anyhow::Result<CPUTensor> {
        self.into_cpu_inner(handle)
    }
}
