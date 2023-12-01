use crate::storage::{CPUStorage, GPUStorage};
use crate::{Shape, Storage};

#[derive(Clone)]
pub struct Tensor<S: Storage> {
    shape: Shape,
    storage: S,
}

pub type CPUTensor = Tensor<CPUStorage>;
pub type GPUTensor = Tensor<GPUStorage>;
