@group(0) @binding(0)
var<storage, read> X: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> S: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> Y: array<vec4<f32>>;

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

fn welford_vcombine(val: vec4<f32>, mean: vec4<f32>, m2: vec4<f32>, count: vec4<f32>) -> mat3x4<f32> {
    let new_count = count + 1.0;
    let delta1 = val - mean;
    let new_mean = mean + delta1 / new_count;
    let delta2 = val - new_mean;
    let new_m2 = m2 + delta1 * delta2; 
    return mat3x4<f32>(new_mean, new_m2, new_count);
}

fn block_welford_combine(b_mean: f32, b_m2: f32, b_count: f32, mean: f32, m2: f32, count: f32) -> vec3<f32> {
    if (b_count == 0.0) {
        return vec3<f32>(mean, m2, count);
    }
    let new_count = count + b_count; 
    let nb_over_n = b_count / new_count;
    let delta = b_mean - mean;
    let new_mean = mean + delta * nb_over_n;
    let new_m2 = m2 + b_m2 + delta * delta * count * nb_over_n;
    return vec3<f32>(new_mean, new_m2, new_count);
}

fn welford_warp_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32) -> vec3<f32> {
    var mean = thread_mean;
    var m2 = thread_m2;
    var count = thread_count;
    for (var offset = subgrp_size >> 1u; offset > 0u; offset >>= 1u) {
        let b_mean = subgroupShuffleDown(mean, offset);
        let b_m2 = subgroupShuffleDown(m2, offset);
        let b_count = subgroupShuffleDown(count, offset);
        let returned = block_welford_combine(b_mean, b_m2, b_count, mean, m2, count);
        mean = returned.x;
        m2 = returned.y;
        count = returned.z;
    }
    return vec3<f32>(mean, m2, count);
}

fn welford_warp_all_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32) -> vec3<f32> {
    let reduced = welford_warp_reduce(thread_mean, thread_m2, thread_count);
    var mean = reduced.x;
    var m2 = reduced.y;
    var count = reduced.z;

    mean = subgroupBroadcast(mean, 0u);
    m2 = subgroupBroadcast(m2, 0u);
    count = subgroupBroadcast(count, 0u);
    return vec3<f32>(mean, m2, count);
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
    let anchor = (group_id.y * metadata.M * metadata.ND4) + group_id.x * metadata.ND4; 
    var threadMean = vec4<f32>(0.0);
    var threadVar = vec4<f32>(0.0);
    var threadCount = vec4<f32>(0.0);
    for (var i = local_id.x; i < metadata.ND4; i+= {{ workgroup_size_x }}u) {
        let a = welford_vcombine(X[anchor + i], threadMean, threadVar, threadCount);
        threadMean = a.x;
        threadVar = a.y;
        threadCount = a.z;
    }
    var finalMean = threadMean.x;
    var finalVar = threadVar.x;
    var finalCount = threadCount.x;
    var returned = block_welford_combine(threadMean.y, threadVar.y, threadCount.y, finalMean, finalVar, finalCount);
    finalMean = returned.x;
    finalVar = returned.y;
    finalCount = returned.z;
    returned = block_welford_combine(threadMean.z, threadVar.z, threadCount.z, finalMean, finalVar, finalCount);
    finalMean = returned.x;
    finalVar = returned.y;
    finalCount = returned.z;
    returned = block_welford_combine(threadMean.w, threadVar.w, threadCount.w, finalMean, finalVar, finalCount);
    finalMean = returned.x;
    finalVar = returned.y;
    finalCount = returned.z;

    let reduced = welford_warp_all_reduce(finalMean, finalVar, finalCount);
    var mean = reduced.x;
    var m2 = reduced.y;
    var count = reduced.z;
    if (subgroup_id == 0u) {
        mu = mean;
        sigma = inverseSqrt(m2 / count + metadata.eps);
    }
    subgroupBarrier();
    for (var i = local_id.x; i < metadata.ND4; i+= {{ workgroup_size_x }}u) {
        let val = X[anchor + i];
        let normalized = (val - vec4<f32>(mu)) * vec4<f32>(sigma);
        Y[anchor + i] = fma(normalized, S[i], B[i]); 
    }
}
