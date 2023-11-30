pub async fn test_harness() {
    let handle = GPUHandle::new().await.unwrap();
    let (M, N, K) = dims;

    check(
        &handle,
        &pipeline,
        workload.count(),
        (M, N, K),
        quantize_b,
        trans_b,
    )
    .await;

    let (A, _) = rand_gpu_buffer::<f32>(&handle, (M, K), false, false);

    let B = match quantize_b {
        Quantization::None => rand_gpu_buffer::<f32>(&handle, (K, N), false, false).0,
        Quantization::SInt8 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::SInt8).0
        }
        Quantization::SInt4 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::SInt4).0
        }
        Quantization::Float16 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::Float16).0
        }
    };

    let (C, _) = rand_gpu_buffer::<f32>(&handle, (M, N), false, true);

    let bind_group_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: A.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: B.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: C.as_entire_binding(),
        },
    ];

    let bind_group = handle
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bind_group_entries,
        });

    let wgc = workload.count();

    //warmup
    let N_WARMUP = 5;
    mm(
        &handle,
        &pipeline,
        &bind_group,
        wgc,
        N_WARMUP,
        &C,
        None,
        dims,
    )
    .await;

    let N_REPEATS = 10;
    let mut profiler = Profiler::new(handle.clone(), N_REPEATS);
    mm(
        &handle,
        &pipeline,
        &bind_group,
        wgc,
        N_REPEATS as _,
        &C,
        Some(&mut profiler),
        dims,
    )
    .await;
}
