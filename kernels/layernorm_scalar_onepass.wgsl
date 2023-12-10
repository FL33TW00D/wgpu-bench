@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read> S: array<f32>;

@group(0) @binding(2)
var<storage, read> B: array<f32>;

@group(0) @binding(3)
var<storage, read_write> Y: array<f32>;

struct Meta {
    M: u32,
    N: u32,
    ND4: u32,
    eps: f32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sum: array<f32, BLOCK_SIZE>;
var<workgroup> sq_sum: array<f32, BLOCK_SIZE>;
var<workgroup> mean: f32;
var<workgroup> sigma: f32;

fn block_reduce(index: u32, stride: u32) {
    if index < stride {
        sum[index] += sum[index + stride];
        sq_sum[index] += sq_sum[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn main(
@builtin(global_invocation_id) global_id: vec3<u32>,
@builtin(local_invocation_id) local_id: vec3<u32>,
@builtin(workgroup_id) group_id: vec3<u32>
) {
    let anchor = (group_id.y * metadata.M * metadata.N) + group_id.x * metadata.N; 

    for (var i = local_id.x; i < metadata.N; i += BLOCK_SIZE) {
       let val = X[anchor + i];
       sum[local_id.x] += val;
       sq_sum[local_id.x] += val * val;
    }
    workgroupBarrier();

    block_reduce(local_id.x, 64u);
    block_reduce(local_id.x, 32u);
    block_reduce(local_id.x, 16u);
    block_reduce(local_id.x, 8u);
    block_reduce(local_id.x, 4u);
    block_reduce(local_id.x, 2u);
    block_reduce(local_id.x, 1u);

    if local_id.x == 0u {
        mean = sum[0] / f32(metadata.N);
        sigma = inverseSqrt(sq_sum[0] / f32(metadata.N) - mean * mean + metadata.eps);
    }
    workgroupBarrier();

    for (var i = local_id.x; i < metadata.N; i += BLOCK_SIZE) {
        let val = (X[anchor + i] - mean) * sigma;
        Y[anchor + i] = fma(val, S[i], B[i]);
    }
}

