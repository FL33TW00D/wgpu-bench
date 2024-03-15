#![allow(non_snake_case)]
use encase::ShaderType;
use inline_python::{python, Context};
use numpy::PyArrayDyn;
use pyo3::Python;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion};
use wgpu_bencher::{
    dispatch_validate, shape, wgc, wgs, CPUTensor, GPUHandle, KernelBench, KernelContextExt,
    OpMetadata, WgpuTimer, Workload,
};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType, derive_new::new, Debug)]
pub struct MatmulMeta {
    aShape: glam::UVec3,
    bShape: glam::UVec3,
    outShape: glam::UVec3,
}

impl OpMetadata for LayerNormMeta {}

#[derive(derive_new::new, Debug)]
pub struct MatmulBenchmark {
    M: usize,
    N: usize,
    K: usize,
}

impl KernelBench for MatmulBenchmark {
    type Metadata = LayerNormMeta;

    fn name() -> &'static str {
        "LayerNorm"
    }

    fn source(workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(Self::name(), include_str!("../../kernels/sgemm/tfjs.wgsl"))
            .unwrap();
        context.insert_workload(workload);
        tera.render(Self::name(), &context).unwrap()
    }

    fn tensors(&self) -> Vec<CPUTensor> {
        let (M, N, K) = (self.M, self.N, self.K);
        let a = CPUTensor::rand::<f32>(shape![M, K]);
        let b = CPUTensor::rand::<f32>(shape![K, N]);
        let output = CPUTensor::zeros::<f32>(shape![M, N]);
        vec![a, b, output]
    }

    fn workload(tensors: &[CPUTensor]) -> Workload {
        let input = &tensors[0];
        let [_B, M, _N] = input.shape().try_into().unwrap();
        Workload::new(wgs![8, 8, 1], wgc![M as _, 1, 1])
    }

    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata {
        let input = &tensors[0];
        let [_B, M, N] = input.shape().try_into().unwrap();
        LayerNormMeta::new(M as _, N as _, (N / 4) as _, self.eps)
    }

    fn validate(&self, tensors: &[CPUTensor]) {
        let (input, scale, bias) = (&tensors[0], &tensors[1], &tensors[2]);
        let ground = Python::with_gil(|py| {
            let (py_input, py_scale, py_bias) = (
                input.to_py::<f32>(&py),
                scale.to_py::<f32>(&py),
                bias.to_py::<f32>(&py),
            );
            let result: Context = python! {
                import torch
                import torch.nn.functional as F

                (input, scale, bias) = (torch.from_numpy('py_input), torch.from_numpy('py_scale), torch.from_numpy('py_bias))
                result = F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self);
        let cpu_result = gpu_tensors.remove(3).into_cpu(TIMER.handle()).unwrap();
        ground.all_close(&cpu_result, 1e-5, 1e-5).unwrap();
    }
}

pub fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    wgpu_bencher::benchmark(
        c,
        &TIMER,
        LayerNorm::new(1e-5),
        PROB_M * PROB_N * std::mem::size_of::<f32>(),
    )
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
