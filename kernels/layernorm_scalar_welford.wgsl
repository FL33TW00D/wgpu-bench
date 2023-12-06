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

fn welford_combine(val: f32, mean: f32, m2: f32, count: f32) -> vec3<f32> {
    let new_count = count + 1.0;
    let delta1 = val - mean;
    let new_mean = mean + delta1 / new_count;
    let delta2 = val - new_mean;
    let new_m2 = m2 + delta1 * delta2;
    return vec3<f32>(new_mean, new_m2, new_count);
}

fn welford_warp_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32) -> vec3<f32> {
    for (var offset = 16u; offset > 0u; offset >>= 1u) {
        let b_mean = subgroupShuffleDown(thread_mean, offset);
        let b_m2 = subgroupShuffleDown(thread_m2, offset);
        let b_count = subgroupShuffleDown(thread_count, offset);
        welford_combine(thread_mean, b_mean, b_m2, b_count);
    }
}

fn welford_warp_all_reduce(thread_mean: f32, thread_m2: f32, thread_count: f32) -> vec3<f32> {
    let reduced = welford_warp_reduce(thread_mean, thread_m2, thread_count);
    var mean = reduced.x;
    var m2 = reduced.y;
    var count = reduced.z;

    mean = subgroupBroadcast(mean, 0);
    m2 = subgroupBroadcast(m2, 0);
    count = subgroupBroadcast(count, 0);
    return vec3<f32>(mean, m2, count);
}


@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let anchor = (group_id.y * metadata.M * metadata.N) + group_id.x * metadata.N; 
    let threadVar = 0f;
    let threadMean = 0f;
    for (var i = local_id.x; i < metadata.N; i+= {{ workgroup_size_x }}u) {
        let returned = welford_combine(X[anchor + i], threadMean, threadVar, threadVar);
    }

    let reduced = welford_warp_all_reduce(threadMean, threadVar, threadVar);
    var mean = reduced.x;
    var m2 = reduced.y;
    var count = reduced.z;
    Y[anchor + local_id.x] = (mean - S[local_id.x]) / sqrt(m2 / count + metadata.eps) + B[local_id.x];
}
