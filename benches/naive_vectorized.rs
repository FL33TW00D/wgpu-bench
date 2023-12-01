#![allow(non_snake_case)]
use encase::ShaderType;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion};
use wgpu_bencher::{
    empty_buffer, rand_gpu_buffer, wgc, wgs, GPUHandle, Kernel, KernelContextExt, OpMetadata,
    Shape, WgpuTimer, Workload,
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

#[derive(derive_new::new)]
pub struct LayerNormProblem {
    X: Shape,
    S: Shape,
    B: Shape,
    Y: Shape,
}

#[derive(derive_new::new, Debug)]
pub struct LayerNorm {
    eps: f32,
}

impl Kernel for LayerNorm {
    type Metadata = LayerNormMeta;
    type Problem = LayerNormProblem;

    fn name() -> &'static str {
        "LayerNorm"
    }

    fn source(workload: Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(
            &Self::name(),
            include_str!("../kernels/layernorm_vec4.wgsl"),
        )
        .unwrap();
        context.insert_workload(&workload);
        tera.render(Self::name(), &context).unwrap()
    }

    fn problem() -> Self::Problem {
        LayerNormProblem::new(
            Shape::new(smallvec![1024, 1024]),
            Shape::new(smallvec![1024]),
            Shape::new(smallvec![1024]),
            Shape::new(smallvec![1024, 1024]),
        )
    }

    fn buffers(&self, handle: &GPUHandle) -> Vec<wgpu::Buffer> {
        let problem = Self::problem();
        let X = rand_gpu_buffer::<f32>(handle, problem.X.numel());
        let S = rand_gpu_buffer::<f32>(handle, problem.S.numel());
        let B = rand_gpu_buffer::<f32>(handle, problem.B.numel());
        let Y = empty_buffer::<f32>(handle.device(), problem.Y.numel());
        vec![X, S, B, Y]
    }

    fn metadata(&self) -> Self::Metadata {
        let problem = Self::problem();
        let N = problem.X[0];
        let M = problem.X[1];
        let ND4 = N / 4;
        LayerNormMeta::new(N as u32, M as u32, ND4 as u32, self.eps)
    }

    fn workload() -> Workload {
        let problem = Self::problem();
        let M = problem.X[1];
        Workload::new(wgs![128 / 4, 1, 1], wgc![M as _, 1, 1])
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
