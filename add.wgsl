@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let M = 1024u;
    let N = 1024u;
    if ((group_id.x * 64u + local_invocation_index) >= M) {
        return;
    }

    let batch_stride = (M * N / 1u);

    let b = f32(B[0]);
    let batch_offset = group_id.y * batch_stride;
    let offset = (group_id.x * 64u + local_invocation_index) * N;

    for(var i = 0u; i < N; i++) {
        let index = batch_offset + offset + i;
        Y[index] = A[index] + b;
    }
}

