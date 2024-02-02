# wgpu-bench

Benchmark any WebGPU Kernel.

Check out `/benches` for an example, simply implement the Kernel trait and boom!

Provide a Python snippet to ensure that your kernel is correct!

## Optimizing a LayerNorm Kernel

Reproduce:
```bash
cat Cargo.toml
cargo bench --bench <bench_name>
``` 
Results on M3 Max 14 core:
```bash
Naive Onepass (precision FAIL)
time:   [88680.6021 ns 92554.9586 ns 95756.6363 ns]
thrpt:  [40.7935 GiB/s 42.2047 GiB/s 44.0485 GiB/s]

Naive
time:   [115698.3828 ns 116433.8667 ns 117343.1857 ns]
thrpt:  [33.2891 GiB/s 33.5491 GiB/s 33.7624 GiB/s]

Naive Vectorized
time:   [113990.1512 ns 114341.8859 ns 114775.5896 ns]
thrpt:  [34.0338 GiB/s 34.1629 GiB/s 34.2683 GiB/s]

Welford Scalar
time:   [74209.2818 ns 74668.5137 ns 75306.7653 ns]
thrpt:  [51.8712 GiB/s 52.3146 GiB/s 52.6383 GiB/s]

Welford Vectorized
time:   [48744.7028 ns 48831.9797 ns 48933.0603 ns]
thrpt:  [79.8284 GiB/s 79.9937 GiB/s 80.1369 GiB/s]
```

## TODO

- [x] Add throughput measurements
- [ ] Encode more commands into a single command buffer (https://github.com/philipturner/metal-flash-attention/issues/12#issuecomment-1850300198)
- [ ] Benchmark comparisons? Shared code between similar kernels?
- [ ] Simplify Kernel trait
- [ ] Cleaning & Polishing 🧽
