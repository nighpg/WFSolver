# Benchmarks

Each case runs in a separate Python process. Peak RSS covers the entire process, including initial imports. `solve_seconds` measures the solver call; `save_seconds` measures NPZ compression and writing. Coefficient evaluation, operator construction, factorization, and linear-solve timings inside integration overlap and must not simply be added. Wall time is not a fixed CI pass/fail criterion.

## Large-grid smoke test

```bash
wf-forward benchmark --grid-points 1001,10001,100001 \
  --methods implicit_euler --cases neutral,selection,mutation \
  --output-counts 10 --duration 1 --dt 0.1 \
  --output benchmarks/acceptance_arm64.json
```

All cases use Ne=10000 and initial frequency 0.3. Selection uses s=0.1, h=0.5; mutation uses u=1e-4, v=2e-4. This short run checks the O(G) implementation, not long-time distribution accuracy. Separate spatial/time convergence tests assess accuracy.

## Compare grids, output counts, and integrators

```bash
wf-forward benchmark --grid-points 1001,10001,100001 \
  --methods expm,implicit_euler --cases neutral,selection,mutation \
  --output-counts 10,100,1000 --duration 0.001 --dt 0.0001 \
  --timeout 120 --output results/full_benchmark.json
```

The full matrix requires substantial storage time and memory. Increase duration separately to measure long-time, large-grid exponential integration. Timed-out cases are explicitly recorded in JSON and cause exit code 3. Compare default threading with `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` prepended to the command. Use `wf-forward inspect <result.npz> --blas` to inspect the current runtime's BLAS configuration.
