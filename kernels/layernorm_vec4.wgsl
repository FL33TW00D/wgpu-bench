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

var<workgroup> smem: array<vec4<f32>, {{ workgroup_size_x }}>; //max 16kb

fn mu(local_id: vec3<u32>, anchor: u32) -> f32 {
    var threadSum = vec4<f32>(0f);
    for (var i: u32 = local_id.x; i < metadata.ND4; i += {{ workgroup_size_x }}u) {
        threadSum += X[anchor + i];
    }
    smem[local_id.x] = threadSum;
    workgroupBarrier();
    
    //Compute μ
    for(var s = {{ workgroup_size_x }}u >> 1u; s > 0u; s >>= 1u) {
        if(local_id.x < s) {
            smem[local_id.x] += smem[local_id.x + s];
        }
        workgroupBarrier();
    }
    let muSum = smem[0].x + smem[0].y + smem[0].z + smem[0].w;
    return muSum / f32(metadata.N);
}

fn sigma(local_id: vec3<u32>, anchor: u32, mu: f32) -> f32 {
    var threadSum = vec4<f32>(0f);
    //Compute σ
    for (var i: u32 = local_id.x; i < metadata.ND4; i += {{ workgroup_size_x }}u) {
        let inner = X[anchor + i] - vec4<f32>(mu);
        threadSum += (inner * inner);
    }
    smem[local_id.x] = threadSum;
    workgroupBarrier();
    
    for(var s = {{ workgroup_size_x }}u >> 1u; s > 0u; s >>= 1u) {
        if(local_id.x < s) {
            smem[local_id.x] += smem[local_id.x + s];
        }
        workgroupBarrier();
    }
    let sigmaSum = smem[0].x + smem[0].y + smem[0].z + smem[0].w;
    return sigmaSum / f32(metadata.N);
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let anchor = (group_id.y * metadata.M * metadata.ND4) + group_id.x * metadata.ND4; 
    let mu = mu(local_id, anchor);
    let sigma = sigma(local_id, anchor, mu);
    let eps = vec4<f32>(metadata.eps);

    let denom = sqrt(vec4<f32>(sigma) + eps);

    for(var i: u32 = local_id.x; i < metadata.ND4; i += {{ workgroup_size_x }}u) {
        let core = (X[anchor + i] - vec4<f32>(mu)) / denom;
        Y[anchor + i] = fma(core, S[i], B[i]);
    }
}
