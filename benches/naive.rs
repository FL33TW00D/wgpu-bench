#![allow(non_snake_case)]
use encase::ShaderType;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion};
use wgpu_bencher::{
    shape, wgc, wgs, CPUTensor, GPUHandle, GPUTensor, Kernel, KernelContextExt, OpMetadata,
    WgpuTimer, Workload,
};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType, derive_new::new, Debug)]
pub struct LayerNormMeta {
    N: u32,
    M: u32,
    ND4: u32,
    eps: f32,
}

impl OpMetadata for LayerNormMeta {}

#[derive(derive_new::new, Debug)]
pub struct LayerNorm {
    eps: f32,
}

impl Kernel for LayerNorm {
    type Metadata = LayerNormMeta;

    fn name() -> &'static str {
        "LayerNorm"
    }

    fn source(workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(
            &Self::name(),
            include_str!("../kernels/layernorm_scalar.wgsl"),
        )
        .unwrap();
        context.insert_workload(workload);
        tera.render(Self::name(), &context).unwrap()
    }

    fn tensors(handle: &GPUHandle) -> Vec<GPUTensor> {
        let input = CPUTensor::rand::<f32>(shape![4, 1024, 1024]).into_gpu(handle);
        let scale = CPUTensor::rand::<f32>(shape![1024]).into_gpu(handle);
        let bias = CPUTensor::rand::<f32>(shape![1024]).into_gpu(handle);
        let output = CPUTensor::zeros::<f32>(shape![4, 1024, 1024]).into_gpu(handle);
        vec![input, scale, bias, output]
    }

    fn workload(tensors: &[GPUTensor]) -> Workload {
        let input = &tensors[0];
        let [_B, _N, M] = input.shape().try_into().unwrap();
        Workload::new(wgs![128, 1, 1], wgc![M as _, 1, 1])
    }

    fn metadata(&self, tensors: &[GPUTensor]) -> Self::Metadata {
        let input = &tensors[0];
        let [_B, N, M] = input.shape().try_into().unwrap();
        LayerNormMeta::new(N as _, M as _, (N / 4) as _, self.eps)
    }
}

pub fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    wgpu_bencher::benchmark(c, &TIMER, LayerNorm::new(1e-5))
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
