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

var<workgroup> mu: f32;
var<workgroup> sigma: f32;
var<workgroup> subgrp_size: u32;

fn welford_combine(val: f32, mean: ptr<function, f32>, m2: ptr<function, f32>, count: ptr<function, f32>) {
    *count += 1.0;
    let delta1 = val - *mean;
    *mean += delta1 / *count;
    let delta2 = val - *mean;
    *m2 += delta1 * delta2;
}

fn block_welford_combine(b_mean: f32, b_m2: f32, b_count: f32, mean: ptr<function, f32>, m2: ptr<function, f32>, count: ptr<function, f32>) { 
    if (b_count == 0.0) {
        return;
    }
    let new_count = *count + b_count; 
    let nb_over_n = b_count / new_count;
    let delta = b_mean - *mean;
    *mean += delta * nb_over_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}

fn welford_warp_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32, mean: ptr<function, f32>, m2: ptr<function, f32>, count: ptr<function, f32>) {
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (var offset = subgrp_size >> 1u; offset > 0u; offset >>= 1u) {
        let b_mean = subgroupShuffleDown(*mean, offset);
        let b_m2 = subgroupShuffleDown(*m2, offset);
        let b_count = subgroupShuffleDown(*count, offset);
        block_welford_combine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

fn welford_warp_all_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32, mean: ptr<function, f32>, m2: ptr<function, f32>, count: ptr<function, f32>) {
    welford_warp_reduce(thread_mean, thread_m2, thread_count, mean, m2, count);

    *mean = subgroupBroadcast(*mean, 0u);
    *m2 = subgroupBroadcast(*m2, 0u);
    *count = subgroupBroadcast(*count, 0u);
}


@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
) {
    subgrp_size = subgroup_size;
    let anchor = (group_id.y * metadata.M * metadata.N) + group_id.x * metadata.N; 
    var threadVar = 0f;
    var threadMean = 0f;
    var threadCount = 0f;
    for (var i = local_id.x; i < metadata.N; i+= {{ workgroup_size_x }}u) {
        welford_combine(X[anchor + i], &threadMean, &threadVar, &threadCount);
    }

    var mean = 0f;
    var m2 = 0f;
    var count = 0f;
    welford_warp_all_reduce(threadMean, threadVar, threadCount, &mean, &m2, &count);

    if (subgroup_id == 0u) {
        mu = mean;
        sigma = inverseSqrt(m2 / count + metadata.eps);
    }
    subgroupBarrier();
    for (var i = local_id.x; i < metadata.N; i+= {{ workgroup_size_x }}u) {
        let val = X[anchor + i];
        let normalized = (val - mu) * sigma;
        Y[anchor + i] = fma(normalized, S[i], B[i]); 
    }
}
