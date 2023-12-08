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

struct VWelford {
    mean: vec4<f32>,
    m2: vec4<f32>,
    count: vec4<f32>,
}

struct Welford {
    mean: f32,
    m2: f32,
    count: f32,
}

fn welford_vcombine(val: vec4<f32>, welford: VWelford) -> VWelford {
    let new_count = welford.count + 1.0;
    let delta1 = val - welford.mean;
    let new_mean = welford.mean + delta1 / new_count;
    let delta2 = val - new_mean;
    let new_m2 = welford.m2 + delta1 * delta2; 
    return VWelford(new_mean, new_m2, new_count); 
}

fn block_welford_combine(b_welford: Welford, welford: Welford) -> Welford {
    if (b_welford.count == 0.0) {
        return welford;
    }
    let new_count = welford.count + b_welford.count; 
    let nb_over_n = b_welford.count / new_count;
    let delta = b_welford.mean - welford.mean;
    let new_mean = welford.mean + delta * nb_over_n;
    let new_m2 = welford.m2 + b_welford.m2 + delta * delta * welford.count * nb_over_n;
    return Welford(new_mean, new_m2, new_count);
}

fn welford_warp_reduce(thread_welford: Welford) -> Welford {
    var welford = thread_welford;
    for (var offset = subgrp_size >> 1u; offset > 0u; offset >>= 1u) {
        let b_mean = subgroupShuffleDown(welford.mean, offset);
        let b_m2 = subgroupShuffleDown(welford.m2, offset);
        let b_count = subgroupShuffleDown(welford.count, offset);
        welford = block_welford_combine(Welford(b_mean, b_m2, b_count), welford);
    }
    return welford;
}

fn welford_warp_all_reduce(thread_welford: Welford) -> Welford {
    var welford = welford_warp_reduce(thread_welford);

    welford.mean = subgroupBroadcast(welford.mean, 0u);
    welford.m2 = subgroupBroadcast(welford.m2, 0u);
    welford.count = subgroupBroadcast(welford.count, 0u);
    return welford; 
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
    var threadWelford = VWelford(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var i = local_id.x; i < metadata.ND4; i+= {{ workgroup_size_x }}u) {
        threadWelford = welford_vcombine(X[anchor + i], threadWelford);
    }
    var scalarWelford = Welford(threadWelford.mean.x, threadWelford.m2.x, threadWelford.count.x);
    scalarWelford = block_welford_combine(Welford(threadWelford.mean.y, threadWelford.m2.y, threadWelford.count.y), scalarWelford);
    scalarWelford = block_welford_combine(Welford(threadWelford.mean.z, threadWelford.m2.z, threadWelford.count.z), scalarWelford);
    scalarWelford = block_welford_combine(Welford(threadWelford.mean.w, threadWelford.m2.w, threadWelford.count.w), scalarWelford);

    let reducedWelford = welford_warp_all_reduce(scalarWelford);
    var mean = reducedWelford.mean; 
    var m2 = reducedWelford.m2;
    var count = reducedWelford.count;
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
