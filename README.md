# wgpu-bench

Benchmark any WebGPU Kernel.

Check out `/benches` for an example, simply implement the Kernel trait and boom!

Provide a Python snippet to ensure that your kernel is correct!

## Optimizing a LayerNorm Kernel
```bash
Naive Onepass
time:   [208725.4932 ns 209048.9067 ns 209391.8682 ns]
thrpt:  [18.6552 GiB/s 18.6858 GiB/s 18.7148 GiB/s]

Naive
time:   [161370.7035 ns 166101.8417 ns 172456.2275 ns]
thrpt:  [22.6507 GiB/s 23.5172 GiB/s 24.2067 GiB/s]

Naive Vectorized
time:   [150533.2318 ns 151013.9824 ns 151519.9951 ns]
thrpt:  [25.7804 GiB/s 25.8668 GiB/s 25.9494 GiB/s]

Welford Scalar
time:   [129993.7867 ns 133295.9256 ns 137187.9144 ns]
thrpt:  [28.4737 GiB/s 29.3051 GiB/s 30.0495 GiB/s]

Welford Vectorized
time:   [111251.7095 ns 113935.1043 ns 117158.9148 ns]
thrpt:  [33.3415 GiB/s 34.2849 GiB/s 35.1118 GiB/s]
```

## TODO

- [ ] Add throughput measurements
- [ ] Benchmark comparisons? Shared code between similar kernels?
- [ ] Simplify Kernel trait
- [ ] Cleaning & Polishing 🧽
