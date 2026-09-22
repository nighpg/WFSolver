# wf-forward

A Wright–Fisher diffusion forward solver for one locus, two alleles, and one diploid population. Requires Python 3.11+, NumPy, and SciPy; the reference backend uses CPU float64.

Time is measured in **generations**, with **V(x,t) = x(1−x)/(2 Ne(t))**. The Fokker–Planck diffusion coefficient is D = V/2. The internal state is probability mass on a grid, including endpoints, rather than density.

## Installation

Use native arm64 Python on Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cli,plot,dev]'
python -c 'import platform; print(platform.machine())'
```

If `.venv` already exists, activate it instead of recreating it.

## Web UI setup and launch

From your local checkout:

```bash
cd /path/to/WFSolver
source .venv/bin/activate
wf-forward serve --open
```

Open **http://127.0.0.1:8765/**. Omit `--open` to open the URL manually. **Opening `web_static/index.html` directly does not enable calculations:** the UI needs the Python API served at this URL. Leave the terminal running; press `Ctrl+C` to stop. If the server is already running, use the same URL. To change ports, use `wf-forward serve --port 8766 --open`.

For a fresh installation, no Node.js or extra web packages are required:

```bash
cd /path/to/WFSolver
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
wf-forward serve --open
```

1. Choose **Neutral**, **Selection**, **Mutation**, or **Bottleneck**, then adjust parameters.
2. Select **Run simulation**. Use **Cancel calculation** to stop a running job.
3. Explore the distribution with the time slider or playback button.
4. Enable **Lock y-axis** to keep the density axis at **0 to Maximum**. It starts with the current displayed maximum; enter any finite positive value. The limit survives time changes, playback, and new simulations until page reload. Disable it to restore automatic scaling. Values above the limit are clipped only in the plot, with an explanatory note.
5. Under **Sampled allele-frequency trajectories**, choose **Paths** and **Random seed**, then select **Sample trajectories**. The chart shows individual paths and the forward-solver mean as a dashed line. The time slider also moves the marker on this chart. Sampling uses the displayed result even if form settings have subsequently changed.
6. Download the configuration, summary CSV, full-distribution NPZ, and sampled paths as CSV or NPZ. Sampling can be repeated with another seed without rerunning the forward solver. Cancelling or failing a new sampling run retains the previous completed sample.

The JSON tab accepts the same configuration as the CLI, including piecewise-constant schedules for all coefficients. Returning to Form uses the form values; JSON edits are not automatically converted. Edited settings mark the existing result as **PREVIOUS RESULT** until the next successful simulation.

The server binds only to `127.0.0.1`; it uses no CDN or cloud service. Web limits are 20,001 grid points, 201 saved times, 200,000 attempted integration steps, 64 MiB of saved arrays, and 120 seconds per calculation. Sampling permits 1–128 paths, seeds from 0 to 4294967295, and at most 5 million jumps. Reduce path count, grid size, or duration if sampling reaches its limit; no partial sample is returned. One calculation runs at a time. Larger jobs can use the Python API or CLI.

Density plots aggregate to at most 801 bins while preserving mass; NPZ files retain the full grid. The latest three simulation jobs and three SFS jobs are retained separately. Simulation jobs and their most recent completed samples are temporary and are removed when the server stops. Download results you want to retain. After updating Python server code, restart the server; after updating only static UI files, reload the page.

## Expected site frequency spectra (SFS)

The **Expected site frequency spectrum** panel calculates neutral demographic expected site counts with continuous mutation input and ancestral equilibrium. It has independent population-history controls and does not use the single-locus initial frequency, mutation drift, or selection settings.

- **PDF symmetric:** equilibrium `g_i = theta*(1/i + 1/(n-i))` at the reference size.
- **Standard derived:** equilibrium `g_i = theta/i` at the reference size.

Enter sampled allele copies `n`, mutation amplitude `theta`, reference population size, and a linear or constant population history. Both conventions are calculated together; switch or overlay them without rerunning. Folded spectra, normalized polymorphic proportions, log/linear axes, a time slider, and CSV/NPZ downloads are available. The symmetric convention is a sum, not an average, and has twice the expected total at the same theta. The neutral demographic extension specifies mirrored mutation sources explicitly; it is not a claim to recover undocumented details of the PDF's original implementation.

```bash
wf-forward sfs examples/sfs_growth.json --output results/sfs_growth.npz --csv results/sfs_growth.csv
```

See [SFS model, conventions, numerical method, and examples](docs/sfs.md). This uses exact neutral sample-moment closure with numerical time integration, avoiding the singular-density endpoint quadrature problem. Population-history inference and SFS selection are outside this version's scope.

## Trajectory sampling

One-time marginal distributions alone do not define a trajectory. The sampler reconstructs the saved model and draws **unconditioned paths of the same finite-grid CTMC**. It starts by drawing from the first saved marginal, then uses exponential waiting times with rate `q_down + q_up` and upward-jump probability `q_up / (q_down + q_up)`. It splits intervals at every coefficient breakpoint. Absorbing endpoints persist; mutation can allow re-entry.

This is exact event-driven sampling for the **discretized CTMC**, subject to floating-point arithmetic. It is not an exact continuous-diffusion or individual-based Wright–Fisher sampler. Sampling variability remains; implicit Euler marginals additionally have time-discretization error. If the first saved time is positive, sampling starts there and does not reconstruct an earlier history. Only observations at saved times are exported; connecting lines do not represent the full jump history. Paths are not conditioned on a final frequency or fixation event.

The saved result must contain a reconstructible scalar or piecewise-constant model. Python callable models cannot be reconstructed and are rejected. A fixed result, path count, seed, observation times, and compatible software environment reproduce the same sample; changing the path count or saved-time grid can change all paths. Metadata records PCG64, seed, NumPy version, event count, start time, source configuration, source integration method, and elapsed time. The sampler does not change the forward result.

```python
from wf_forward import SimulationResult

result = SimulationResult.load_npz('results/example.npz')
paths = result.sample_trajectories(paths=32, seed=42)
# paths.times: (T,); paths.frequencies: (32, T)
paths.save_npz('results/paths.npz')
paths.to_csv('results/paths.csv')
```

Trajectory NPZ contains `times`, `frequencies`, and `metadata_json`; load with `numpy.load(..., allow_pickle=False)`. CSV columns are `path_id,time,allele_frequency`, with zero-based path IDs. The API/CLI default event budget is 5 million and the output-array limit is 256 MiB. `max_events` / `--max-events` can increase the event budget deliberately.

## Python API

```python
from wf_forward import (
    WrightFisherModel, UniformGrid, DeltaInitialCondition,
    SolverConfig, ForwardSolver, SimulationResult,
)

model = WrightFisherModel(Ne=1000, selection=0.002, dominance=0.5)
solver = ForwardSolver(
    model,
    UniformGrid(points=501),  # Includes endpoints; G = points - 1 intervals.
    SolverConfig(output_times=[0, 100, 500, 1000]),
)
result = solver.solve(DeltaInitialCondition(x0=0.3))
print(result.mean(), result.p_loss, result.p_fix)
result.save_npz('results/example.npz')
result.to_summary_csv('results/example.csv')
loaded = SimulationResult.load_npz('results/example.npz')
```

`method="auto"` selects `expm` for time-homogeneous models and `implicit_euler` for time-dependent models. Fitness conventions are `w_aa=1, w_Aa=1+h*s, w_AA=1+s`, with `M_sel=s*x*(1-x)*(h+(1-2*h)*x)`. Thus `h=0.5` gives drift `s*x*(1-x)/2`.

## CLI

```bash
wf-forward validate examples/neutral.yaml
wf-forward run examples/neutral.yaml --output results/neutral.npz
wf-forward inspect results/neutral.npz
wf-forward inspect results/neutral.npz --blas
wf-forward plot results/neutral.npz --output results/neutral.png
wf-forward sample results/neutral.npz --paths 32 --seed 42 \
  --output results/paths.npz --csv results/paths.csv
wf-forward run examples/bottleneck.yaml --output results/bottleneck.npz
wf-forward run examples/mutation.yaml --output results/mutation.npz
```

`python -m wf_forward` exposes the same CLI. Logs go to stderr and inspection JSON goes to stdout. Put `--log-level ERROR|WARNING|INFO|DEBUG` before the subcommand. Inputs can be YAML or JSON; paths are relative to the working directory. Unknown settings and schemas are rejected. YAML never loads Python code.

Exit codes: success 0; configuration/input error 2; numerical error 3; I/O error 4; internal error 5.

## Time-dependent models

```python
from wf_forward import PiecewiseConstant

model = WrightFisherModel(
    Ne=PiecewiseConstant([0, 500, 800], [10000, 1000, 10000]),
)
config = SolverConfig(
    output_times=[0, 100, 500, 600, 800, 1000],
    method='implicit_euler',
    dt=0.5,
)
```

Schedules are right-continuous, start at time 0, and retain their last value thereafter. Steps split at saved times and the union of all coefficient breakpoints. Initial conditions are always at t=0, even if the first requested output is later. Unsorted or duplicate times are rejected, not reordered.

Without `dt`, backward Euler uses midpoint coefficients and step-doubling: compare one full step with two half steps in L1 and accept the two-half-step solution. `atol` / `rtol` control a local error indicator, not guaranteed final-distribution error. Limits include `min_dt`, `max_dt`, `max_steps`, and `max_rejections`. The initial step is one tenth of the output spacing, limited to one twentieth of `parameter_timescale` when supplied.

The API accepts `f(t)` for coefficients and `f(x,t)` for `drift_override`, which replaces the entire drift, including selection and mutation. Functions must be deterministic and side-effect-free; drift must return a finite array matching the frequency array. Arbitrary jumps and rapid oscillations cannot always be detected: supply `breakpoints` and a suitable step size. Callables are considered time-dependent unless the user explicitly guarantees `time_homogeneous=True`.

## Initial conditions and endpoint semantics

Available initial conditions: `DeltaInitialCondition`, `ArrayInitialCondition`, `DensityInitialCondition`, and `BetaInitialCondition`. Arrays represent mass over the full grid; density arrays include endpoint nodes. Densities are projected onto nodal mass by integrating piecewise-linear basis functions; Beta projection uses incomplete Beta functions. Nonunit input mass is rejected unless array/density input explicitly sets `normalize=True`.

Delta allocation and density projection preserve total mass and mean, but cannot resolve frequencies below the grid spacing. Near-boundary density can allocate initial mass to endpoints, causing initial absorption error in absorbing models. Diagnostics record this mass and issue resolution warnings. For example, a new mutation with Ne=10000 has x0=1/(2 Ne); the recommendation dx<=x0/2 requires at least 40001 points.

- `boundary="absorbing"`: endpoints absorb; `p_loss` / `p_fix` are loss/fixation probabilities.
- `boundary="mutation"`: endpoints may release mass; `p_at_zero` / `p_at_one` are discrete-grid occupancy probabilities. Compatibility aliases `p_loss` / `p_fix` are **not cumulative absorption probabilities** here.
- Mutation CSV uses `p_at_zero` / `p_at_one`, and distribution CSV uses endpoint states `at_zero` / `at_one`.
- `density` contains interior nodes only. Means and variances include endpoint mass.

Under neutrality and bidirectional mutation, the continuous stationary density is `Beta(4*Ne*u, 4*Ne*v)`. Finite-grid endpoint occupancy must not be interpreted as endpoint atoms of that continuous distribution.

## Accuracy, performance, and reproducibility

The generator has nonnegative off-diagonals and zero row sums, using first-order upwind drift and centered diffusion. Interior Péclet numbers above 2 trigger a refinement warning. Implicit integration uses tridiagonal LAPACK, caching factorizations for identical operators and step sizes and reusing work buffers. SciPy does not expose internal exponential-action step counts: `steps=null` and API call counts are reported separately.

Probability checks run **before correction**. By default only actual negative entries down to −1e-14 are clipped, with renormalization only following such correction. Larger negatives fail. `negative_policy="warn"` allows warned correction down to `negative_error_tolerance`. Positive arrays are not routinely renormalized.

Operator and integration workspace are O(G); saved results are O(TG). `max_output_bytes` limits saved arrays only, defaults to 512 MiB, and is not a process memory cap. 1000 times × 100001 points requires about 800 MB for the saved mass alone. Large, long-duration `expm` runs may be expensive: compare with time-converged implicit integration. The method never changes silently mid-run.

NPZ loading disables pickle; saving uses atomic temporary-file replacement. Results contain normalized configuration, initial condition, environment, selected method, timing, pre-correction conservation errors, and correction history. Declarative runs can be reconstructed through `wf_forward.cli.from_config(result.metadata['config'])`. Callable names, labels, and source hashes are recorded where available, but code and closures are not restorable.

## Tests and benchmarks

```bash
python -m pytest -q
wf-forward benchmark --grid-points 1001,10001,100001 --output results/benchmark.json
```

Tests cover generator identities, dense exponential references, neutral fixation and moments, multiple-seed Monte Carlo, spatial/time convergence, mutation stationarity, persistence/CLI, a 100000-interval implicit solve, Web API behavior, and sampled-path temporal dependence, marginals, schedules, reproducibility, and limits.

Benchmark cases run in separate processes and record peak RSS and environment. See [benchmark instructions](benchmarks/README.md) and [validation results](benchmarks/validation.md).

The first release supports generations, uniform grids, and float64. Scaled time, Crank–Nicolson, Chang–Cooper, GPU, nonuniform grids, streaming, and parallel parameter sweeps remain future work; unsupported options fail explicitly.
