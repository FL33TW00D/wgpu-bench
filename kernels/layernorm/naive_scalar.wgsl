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

var<workgroup> smem: array<f32, BLOCK_SIZE>; //max 16kb

fn block_sum(index: u32, stride: u32) {
    if index < stride {
        smem[index] += smem[index + stride];
    }
    workgroupBarrier();
}

fn mu(local_id: vec3<u32>, anchor: u32) -> f32 {
    var threadSum = 0f;
    for (var i: u32 = local_id.x; i < metadata.N; i += BLOCK_SIZE) {
        threadSum += X[anchor + i];
    }
    smem[local_id.x] = threadSum;
    workgroupBarrier();
    
    block_sum(local_id.x, 64u);
    block_sum(local_id.x, 32u);
    block_sum(local_id.x, 16u);
    block_sum(local_id.x, 8u);
    block_sum(local_id.x, 4u);
    block_sum(local_id.x, 2u);
    block_sum(local_id.x, 1u);

    return smem[0] / f32(metadata.N); 
}

fn sigma(local_id: vec3<u32>, anchor: u32, mu: f32) -> f32 {
    var threadSum = 0f;
    //Compute σ
    for (var i: u32 = local_id.x; i < metadata.N; i += BLOCK_SIZE) {
        let val = X[anchor + i] - mu;
        threadSum += (val * val);
    }
    smem[local_id.x] = threadSum;
    workgroupBarrier();
    
    block_sum(local_id.x, 64u);
    block_sum(local_id.x, 32u);
    block_sum(local_id.x, 16u);
    block_sum(local_id.x, 8u);
    block_sum(local_id.x, 4u);
    block_sum(local_id.x, 2u);
    block_sum(local_id.x, 1u);

    return smem[0] / (f32(metadata.N));
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let anchor = (group_id.y * metadata.M * metadata.N) + group_id.x * metadata.N; 
    let mu = mu(local_id, anchor);
    let sigma = sigma(local_id, anchor, mu);

    let denom = inverseSqrt(sigma + metadata.eps);

    for(var i: u32 = local_id.x; i < metadata.N; i += BLOCK_SIZE) {
        let core = (X[anchor + i] - mu) * denom;
        Y[anchor + i] = fma(core, S[i], B[i]); 
    }
}
