# wgpu-bench

Benchmark any WebGPU Kernel.

Check out `/benches` for an example, simply implement the Kernel trait and boom!

Provide a Python snippet to ensure that your kernel is correct!

```bash
LayerNormScalar     time:   [149355.4133 ns 156099.6010 ns 164338.7318 ns]
                    change: [-0.5438% +3.9567% +8.4847%] (p = 0.09 > 0.05)
LayerNormVectorized time:   [144526.6691 ns 145412.3884 ns 146351.1508 ns]
                    change: [-6.1644% -2.2516% +1.3739%] (p = 0.26 > 0.05)
```
