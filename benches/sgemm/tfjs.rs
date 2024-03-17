#![allow(non_snake_case)]
use encase::ShaderType;
use inline_python::{python, Context};
use numpy::PyArrayDyn;
use pyo3::Python;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
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
pub struct SGEMMMeta {
    aShape: glam::IVec3,
    aStrides: glam::IVec3,
    bShape: glam::IVec3,
    bStrides: glam::IVec3,
    outShape: glam::IVec3,
    outStrides: glam::IVec3,
    dimInner: i32,
}

impl OpMetadata for SGEMMMeta {}

#[derive(derive_new::new, Debug)]
pub struct SGEMMBenchmark {
    B: usize,
    M: usize,
    N: usize,
    K: usize,
    TILE_DIM: usize,
    ROW_PER_THREAD: usize,
}

impl SGEMMBenchmark {
    fn shape_fit(&self) -> [bool; 3] {
        let aOuter = self.M;
        let bOuter = self.N;
        let dimInner = self.K;

        let mut shape_fit = [false; 3];
        shape_fit[0] = aOuter % self.TILE_DIM == 0;
        shape_fit[1] = bOuter % self.TILE_DIM == 0;
        shape_fit[2] = dimInner % self.TILE_DIM == 0;
        println!("SHAPE FIT: {:?}", shape_fit);
        shape_fit
    }
}

impl KernelBench for SGEMMBenchmark {
    type Metadata = SGEMMMeta;

    fn name() -> &'static str {
        "SGEMMBenchmark"
    }

    fn source(&self, workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        let is_vec4 = (self.M % 4 == 0) && (self.N % 4 == 0) && (self.K % 4 == 0);
        let template = if is_vec4 {
            include_str!("../../kernels/sgemm/tfjs.wgsl")
        } else {
            include_str!("../../kernels/sgemm/scalar_tf.wgsl")
        };
        tera.add_raw_template(Self::name(), template).unwrap();
        let shape_fit = self.shape_fit();
        context.insert("A_FIT", &shape_fit[0]);
        context.insert("B_FIT", &shape_fit[1]);
        context.insert("OUT_FIT", &shape_fit[2]);

        context.insert("TILE_DIM", &self.TILE_DIM);
        context.insert("ROW_PER_THREAD", &self.ROW_PER_THREAD);
        context.insert_workload(workload);
        let kernel = tera.render(Self::name(), &context).unwrap();
        println!("{}", kernel);
        kernel
    }

    fn tensors(&self) -> Vec<CPUTensor> {
        let (B, M, N, K) = (self.B, self.M, self.N, self.K);
        let a = CPUTensor::randn::<f32>(shape![B, M, K]);
        let b = CPUTensor::randn::<f32>(shape![B, K, N]);
        let output = CPUTensor::zeros::<f32>(shape![B, M, N]);
        vec![a, b, output]
    }

    fn workload(&self, _: &[CPUTensor]) -> Workload {
        let (TILE_DIM, ROW_PER_THREAD) = (self.TILE_DIM, self.ROW_PER_THREAD);
        let workgroup_size = wgs![(TILE_DIM / 4) as _, (TILE_DIM / ROW_PER_THREAD) as _, 1];
        let group_x = Workload::ceil(self.N, TILE_DIM);
        let group_y = Workload::ceil(self.M, TILE_DIM);
        let workgroup_count = wgc![group_x as _, group_y as _, self.B as u32];
        let dispatch = Workload::new(workgroup_size, workgroup_count);
        println!("DISPATCH: {:?}", dispatch);
        dispatch
    }

    fn metadata(&self, _: &[CPUTensor]) -> Self::Metadata {
        let (B, M, N, K) = (self.B as i32, self.M as i32, self.N as i32, self.K as i32);

        let aShape = glam::IVec3::new(B, M, K);
        let aStrides = glam::IVec3::new(M * K, K, 1);
        let bShape = glam::IVec3::new(B, K, N);
        let bStrides = glam::IVec3::new(K * N, N, 1);
        let outShape = glam::IVec3::new(B, M, N);
        let outStrides = glam::IVec3::new(M * N, N, 1);

        let meta = SGEMMMeta::new(aShape, aStrides, bShape, bStrides, outShape, outStrides, K);
        println!("META: {:?}", meta);
        meta
    }

    fn validate(&self, tensors: &[CPUTensor]) {
        let (a, b) = (&tensors[0], &tensors[1]);
        let ground = Python::with_gil(|py| {
            let (py_a, py_b) = (a.to_py::<f32>(&py), b.to_py::<f32>(&py));
            let result: Context = python! {
                import torch
                (a, b) = (torch.from_numpy('py_a), torch.from_numpy('py_b))
                result = (a @ b).numpy()
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self);
        let cpu_result = gpu_tensors.remove(2).into_cpu(TIMER.handle()).unwrap();
        println!("GROUND: {}", ground);
        println!("OURS: {}", cpu_result);
        ground.all_close(&cpu_result, 1e-5, 1e-5).unwrap();
    }
}

pub fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let B = 1;
    let M = 2047;
    let N = 2048;
    let K = 2048;
    let TILE_DIM = 32;
    let ROW_PER_THREAD = 4;
    let bench = SGEMMBenchmark::new(B, M, N, K, TILE_DIM, ROW_PER_THREAD);
    let throughput = Throughput::Elements(2 * (B * M * N * K) as u64);
    wgpu_bencher::benchmark(c, &TIMER, bench, throughput)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
