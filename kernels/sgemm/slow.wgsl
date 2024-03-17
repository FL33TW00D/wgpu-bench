//Unoptimized, only gets 500GFLOP
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

struct Meta {
    aShape: vec3<i32>,
    aStrides: vec3<i32>,
    bShape: vec3<i32>,
    bStrides: vec3<i32>,
    outShape: vec3<i32>,
    outStrides: vec3<i32>,
    dimInner: i32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = u32(metadata.aShape.y);
    let N = u32(metadata.bShape.z);
    let K = u32(metadata.aShape.z);

    let a_offset = global_id.z * u32(metadata.aStrides.x);
    let b_offset = global_id.z * u32(metadata.bStrides.x);
    let c_offset = global_id.z * u32(metadata.outStrides.x);

    let cRow = global_id.x;
    let cCol = global_id.y;  
    if (cRow < M && cCol < N / 4u) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < K / 4u; k++) {
          let a = A[a_offset + (cRow * (K / 4u) + k)];
          let b_step = k * N + cCol; //4 rows per iter
          let b_stride = N / 4u;

          tmp = fma(vec4<f32>(a.x), B[b_offset + b_step], tmp); 
          tmp = fma(vec4<f32>(a.y), B[b_offset + (b_step + b_stride)], tmp);
          tmp = fma(vec4<f32>(a.z), B[b_offset + (b_step + (2u * b_stride))], tmp);
          tmp = fma(vec4<f32>(a.w), B[b_offset + (b_step + (3u * b_stride))], tmp);
        }
        C[c_offset + (cRow * (N / 4u) + cCol)] = tmp; 
    }
}

