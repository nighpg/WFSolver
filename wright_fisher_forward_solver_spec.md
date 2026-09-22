# Wright–Fisher diffusion forward solver for Apple Silicon: implementation specification

## 1. Document information

| Item | Description |
|---|---|
| Scope | One locus, two alleles, one population; Wright–Fisher diffusion |
| Uses | Population-genetics research, method validation, parameter sweeps, education |
| Language | Python 3.11 or later |
| Primary platform | macOS / Apple Silicon (arm64) |
| Dependencies | NumPy, SciPy, Matplotlib, PyYAML for CLI configuration, pytest |
| First-release scope | Time-dependent selection, mutation, and effective population size; absorbing and mutation boundaries; sparse solvers |

MUST, SHOULD, and MAY mean required, strongly recommended, and optional, respectively.

This is the English edition of the original implementation specification. It retains the original scope, including optional and future features; it is not a claim that every proposed option is implemented. See the current README for supported behavior and `implementation_review.md` for clarifications adopted during development. In particular, the implementation uses `method="auto"`, defines density projection and mutation endpoint semantics explicitly, and includes basic step-doubling in the first release. The Web UI and trajectory extension are described in the README.

## 2. Purpose

Evolve the distribution of allele frequency x in [0,1] under the Wright–Fisher forward equation and reproducibly calculate:

- Interior probability density phi(x,t).
- Allele loss and fixation probabilities.
- Mean, variance, heterozygosity, and related summaries.
- Saved distributions at requested times for visualization and reuse.

Research quality requirements are probability conservation (interior plus boundaries sums to one), nonnegativity within numerical tolerance, spatial/time convergence to known or refined solutions, reproducible configuration/software/numerics, and interchangeable coefficient functions for selection, mutation, Ne(t), and future migration.

### 2.1 Non-goals

The first release excludes linked multilocus models, direct grids for joint frequency spectra in two or more dimensions, a full individual-based Wright–Fisher simulator, Bayesian/maximum-likelihood inference or gradients, and any requirement for GPU execution.

## 3. Mathematical model

### 3.1 Forward equation

Let X_t=x denote allele A frequency and phi(x,t) the interior density. Use the forward Kolmogorov / Fokker–Planck equation

\[
\partial_t\phi=-\partial_x(M\phi)+\tfrac12\partial_x^2(V\phi).
\]

The corresponding Itô equation is

\[
dX_t=M(X_t,t)\,dt+\sqrt{V(X_t,t)}\,dW_t.
\]

The default time unit is **generations**.

### 3.2 Variance coefficient

For diploid effective population size Ne(t),

\[
V(x,t)=\frac{x(1-x)}{2N_e(t)},\qquad D(x,t)=V(x,t)/2=\frac{x(1-x)}{4N_e(t)}.
\]

The public method `variance(x,t)` returns V. Avoid an ambiguous public `diffusion()` method that could mean either V or D.

### 3.3 Deterministic drift

The default drift is M=M_sel+M_mut.

#### Selection

With genotype fitnesses w_aa=1, w_Aa=1+h*s, w_AA=1+s, use the weak-selection approximation

\[
M_{\rm sel}(x,t)=s(t)x(1-x)\{h(t)+(1-2h(t))x\}.
\]

Additive selection has h=1/2. Permit replacement of M by a user-defined function.

#### Bidirectional mutation

For forward rate u(t), a→A, and backward rate v(t), A→a,

\[
M_{\rm mut}(x,t)=u(t)(1-x)-v(t)x.
\]

### 3.4 Time scaling

Internal calculations default to generations. Scaled time such as tau=t/(2*Ne_ref) MAY be provided. If supported, always record `time_unit` (`generation` or `scaled`), `reference_Ne` for scaled time, and the units of M and V. Never convert time units implicitly inside the API.

### 3.5 Observables

For interior mass p_i and nodes x_i, provide at least

\[
E[X_t]=P_{\rm fix}(t)+\sum_{i=1}^{G-1}x_i p_i(t),
\qquad E[2X_t(1-X_t)]=\sum_{i=1}^{G-1}2x_i(1-x_i)p_i(t).
\]

Never omit endpoint contributions to the mean or other statistics. In mutation mode, the endpoint term is occupancy at x=1 rather than cumulative fixation.

## 4. Numerical methods

### 4.1 Design

The standard method conservatively discretizes space as a continuous-time Markov chain (CTMC), then solves

\[
\frac{d\mathbf p}{dt}=Q(t)^\mathsf{T}\mathbf p(t).
\]

The state p contains **probability mass**, not density. This explicitly represents endpoint states, enables conservation and positivity through a valid generator, uses efficient one-dimensional tridiagonal sparse structure, and allows exponential action for time-homogeneous coefficients without time-stepping truncation error.

### 4.2 Spatial grid

A uniform grid is required initially:

\[
x_i=i\Delta x,\quad i=0,\ldots,G,\qquad \Delta x=1/G.
\]

`grid_points=G+1` includes endpoints; there are G−1 interior nodes. Default: 2001 points. Require at least 101 points. Retain an explicit float64 x array for future nonuniform grids; the v1 generator MAY reject nonuniform grids.

### 4.3 Generator construction

At interior nodes,

\[
q_{i,i+1}=\frac{V_i}{2\Delta x^2}+\frac{\max(M_i,0)}{\Delta x},
\qquad q_{i,i-1}=\frac{V_i}{2\Delta x^2}+\frac{\max(-M_i,0)}{\Delta x},
\]

\[
q_{i,i}=-(q_{i,i-1}+q_{i,i+1}).
\]

This combines centered diffusion with first-order upwind advection. Off-diagonals are nonnegative and row sums vanish.

- `Q[i,j]` is the transition rate i→j, using the backward-generator row convention.
- The forward equation for column mass is `dp_dt = Q.T @ p`.
- Use `scipy.sparse.diags` for sparse construction and CSC/CSR for computation.
- float64 is the default validated precision. float32 is experimental, never the research default.

Diffusion is second-order in space; advection is first-order. Report interior local Péclet numbers

\[
\mathrm{Pe}_i=|M_i|\Delta x/D_i,\qquad D_i=V_i/2.
\]

Warn that the grid should be refined if max(Pe)>2.

### 4.4 Higher-accuracy options

Version 1.1 or later MAY add Chang–Cooper or Scharfetter–Gummel fluxes as `spatial_scheme="chang_cooper"`. Such schemes must preserve positivity/conservation, improve stationary mutation-distribution error over upwind, approach centered differences continuously as |Pe|→0, and evaluate Bernoulli functions stably without overflow at large |Pe|.

### 4.5 Time integration

#### Time-homogeneous coefficients: sparse exponential action

For constant Ne,s,h,u,v,

\[
\mathbf p(t_k)=\exp(Q^\mathsf{T}t_k)\mathbf p(0).
\]

Use `scipy.sparse.linalg.expm_multiply` as the default homogeneous method (`method="expm"`). Prefer one start/stop/num call for equally spaced outputs; advance interval by interval for irregular times. Only tiny roundoff negatives may be corrected under §7.4.

#### Time-dependent coefficients: backward Euler

Evaluate Q at each interval midpoint and solve

\[
(I-\Delta t Q(t_{n+1/2})^\mathsf{T})\mathbf p_{n+1}=\mathbf p_n.
\]

Use `method="implicit_euler"` by default for time-dependent coefficients. Prefer `solve_banded` or a dedicated Thomas algorithm for tridiagonal grids; use sparse `spsolve` for general sparse matrices. Reuse factorizations for identical Q and dt. Backward Euler is first-order in time; positivity takes priority.

#### Optional Crank–Nicolson

\[
(I-\tfrac{\Delta t}{2}Q_{n+1/2}^\mathsf{T})\mathbf p_{n+1}
=(I+\tfrac{\Delta t}{2}Q_{n+1/2}^\mathsf{T})\mathbf p_n.
\]

This MAY be offered as `method="crank_nicolson"`, with mandatory negativity checks because large steps need not preserve positivity. It must not be the research default.

### 4.6 Time-step control

Permit user-specified dt. Otherwise initialize with

\[
\Delta t_{\rm initial}=\min(\Delta t_{\rm output,min}/10,\Delta t_{\rm parameter}/20).
\]

The parameter timescale is the shortest significant relative variation timescale among the coefficients. Also limit steps by the next piecewise-constant breakpoint. Do not impose an explicit stability restriction based on max(−q_ii) as a mandatory implicit step size.

Basic adaptive step-doubling MAY:

1. Compare one full step with two half steps.
2. Accept when their L1 difference is at most atol+rtol*||p||_1.
3. Halve rejected steps; grow subsequent steps by at most two when error is sufficiently small.

Implementation clarification: arbitrary callable timescales cannot generally be inferred; allow an explicit `parameter_timescale`, include dominance variation, accept the two-half-step solution, and bound steps/rejections.

### 4.7 Initial conditions

Support delta(x0), arrays of grid mass, function/array densities integrated into mass, and Beta(alpha,beta).

Delta allocation distributes mass linearly to the neighboring nodes so its mean is x0. Exact endpoints use their boundary state. For a new mutation x0=1/(2*Ne), recommend dx<=x0/2 and warn otherwise.

Validate positive total input mass. By default require total mass within 1±1e-12; renormalize only when explicitly requested with `normalize=True`.

## 5. Boundary conditions

### 5.1 Absorption

Without mutation, endpoints 0 and 1 absorb: their generator rows are zero. Preserve transitions from node 1 into 0 and node G−1 into G, accumulating

\[
P_{\rm loss}=p_0,\qquad P_{\rm fix}=p_G,\qquad p_0+\sum_{i=1}^{G-1}p_i+p_G=1.
\]

### 5.2 Mutation

With positive bidirectional mutation, endpoints are not absorbing. Under `boundary="mutation"`, allow inward drift rates

\[
q_{0,1}=M(0,t)/\Delta x\quad\text{when }M(0,t)>0,
\qquad q_{G,G-1}=-M(1,t)/\Delta x\quad\text{when }M(1,t)<0.
\]

Outward transitions are always zero. Reject absorbing mode with positive u or v by default. Warn when mutation mode has u=v=0, allowing the mathematically absorbing result. Reject user-defined outward endpoint drift.

Implementation clarification: mutation endpoint values are occupancy, not cumulative loss/fixation; expose `p_at_zero` and `p_at_one` and label exports accordingly.

### 5.3 Density output

Define interior density phi_i=p_i/dx for i=1,...,G−1. Never mix endpoint mass into density. Return interior density and endpoint probabilities separately.

## 6. Software structure

Recommended package: `wf_forward`.

```text
wf-forward/
├── pyproject.toml
├── README.md
├── src/wf_forward/
│   ├── __init__.py
│   ├── config.py          # Configuration and validation
│   ├── grid.py            # Grids and initial conditions
│   ├── coefficients.py    # M, V, Ne, selection, mutation
│   ├── operator.py        # Generator construction
│   ├── integrators.py     # Exponential and implicit integration
│   ├── solver.py          # Public solver API
│   ├── result.py          # Results, statistics, persistence
│   ├── diagnostics.py     # Conservation, positivity, Peclet numbers
│   ├── io.py              # NPZ, CSV, JSON/YAML
│   ├── plotting.py        # Visualization
│   └── cli.py             # wf-forward entry point
├── tests/{unit,integration,convergence,regression}/
├── benchmarks/
└── examples/
```

The original proposal separates core, plot, and dev dependency groups. The implementation installs NumPy/SciPy directly, with plot, cli, and dev extras; see README.

## 7. Python API

### 7.1 Public API

```python
from wf_forward import (
    WrightFisherModel, UniformGrid, DeltaInitialCondition,
    SolverConfig, ForwardSolver, SimulationResult,
)

model = WrightFisherModel(
    Ne=10_000, selection=0.01, dominance=0.5,
    mutation_forward=0.0, mutation_backward=0.0, boundary="absorbing",
)
grid = UniformGrid(points=20_001)
initial = DeltaInitialCondition(x0=0.05)
config = SolverConfig(
    method="expm", output_times=[0, 100, 500, 1000],
    probability_tolerance=1e-10,
)
result = ForwardSolver(model, grid, config).solve(initial)
```

### 7.2 Coefficient types

Ne, selection, dominance, mutation_forward, and mutation_backward accept scalars or `Callable[[float],float]`. Expose `drift(x,t)`, `variance(x,t)`, and `is_time_homogeneous()`.

Python callables are not serializable. Record their qualified name, user label, and source hash when available. The initial CLI accepts declarative piecewise-constant schedules and must not load callables by executing arbitrary code.

### 7.3 Result object

`SimulationResult` is a frozen dataclass with:

| Attribute | Shape / meaning |
|---|---|
| times | float64 (T,) |
| x | float64 (G+1,) |
| probability_mass | float64 (T,G+1) |
| metadata | Mapping of names to configuration/environment information |
| diagnostics | Mapping of names to arrays, numbers, or strings |
| density | float64 (T,G−1), interior only |
| p_loss, p_fix, p_segregating | Time series of endpoint/interior probability |

Provide mean(), variance(), heterozygosity(), save_npz(path), and to_summary_csv(path). Probability mass is the canonical, least-lossy representation. Add per-time streaming if needed for memory savings.

### 7.4 Errors and warnings

Provide `ConfigurationError` for invalid parameters/times/boundaries, `NumericalError` for excessive negativity/conservation or linear-solve failures, `ResolutionWarning` for inadequate spatial/time resolution, and `ModelConsistencyWarning` for potential model/boundary mismatch.

Default negative-value policy:

- Actual tiny negatives with min(p)>=−1e-14 may be clipped and renormalized, with diagnostics.
- −1e-10<=min(p)<−1e-14: configurable warning/correction or error; default error.
- min(p)<−1e-10: always a numerical error under the default thresholds.

Thresholds are configurable and must be stored in metadata. Clarification: check total mass before correction and do not renormalize an already nonnegative array routinely.

## 8. CLI

### 8.1 Commands

```bash
wf-forward run config.yaml --output results/run_001.npz
wf-forward inspect results/run_001.npz
wf-forward plot results/run_001.npz --output results/run_001.png
wf-forward validate config.yaml
wf-forward benchmark --grid-points 1001,10001,100001
```

### 8.2 Configuration example

```yaml
schema_version: 1
model:
  ploidy: 2
  Ne: 10000
  selection: 0.01
  dominance: 0.5
  mutation_forward: 0.0
  mutation_backward: 0.0
  boundary: absorbing
  time_unit: generation
grid:
  type: uniform
  points: 20001
initial_condition:
  type: delta
  x0: 0.05
solver:
  spatial_scheme: upwind_ctmc
  method: expm
  output_times: [0, 100, 500, 1000]
  dtype: float64
  probability_tolerance: 1.0e-10
output:
  format: npz
  include_full_distribution: true
  summary_csv: results/run_001_summary.csv
```

### 8.3 Time-dependent Ne

```yaml
model:
  Ne:
    type: piecewise_constant
    breakpoints: [0, 500, 800]
    values: [10000, 1000, 10000]
  selection: 0.0
  dominance: 0.5
  mutation_forward: 0.0
  mutation_backward: 0.0
  boundary: absorbing
solver:
  method: implicit_euler
  dt: 0.5
  output_times: [0, 100, 500, 600, 800, 1000]
```

Always split integration at breakpoints; no step may cross a discontinuity.

### 8.4 Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 2 | Configuration/input error |
| 3 | Numerical error |
| 4 | I/O error |
| 5 | Internal error |

## 9. Data structures and persistence

### 9.1 Memory representation

Use float64 for x (G+1,), current p (G+1,), sparse Q (G+1,G+1), strictly increasing nonduplicate times (T,), and saved P (T,G+1). Retain only requested outputs. Production code MUST NOT construct dense Q; the G=100000 operator/integration path must use O(G) workspace.

### 9.2 NPZ

Compressed NPZ is the standard artifact, containing times, x, probability_mass, metadata_json (UTF-8 JSON), and diagnostics_json. Metadata includes schema version, normalized input, package/Python/NumPy/SciPy versions, macOS version and architecture, start time and duration, Git commit when available, integrator and actual step counts/minimum/maximum dt, maximum corrected negative magnitude and correction count, and maximum conservation error.

### 9.3 CSV

Use CSV for summaries or a single-time distribution, not as the standard large full-history format.

```text
time,p_loss,p_fix,p_segregating,mean,variance,heterozygosity,total_probability
time,x,probability_mass,density,state
```

Absorbing states are labeled loss/interior/fixation; endpoint density is blank. Mutation exports use occupancy names as clarified in the review and README.

## 10. Validation

### 10.1 Analytical properties

Neutral mean conservation: E[X_t]=x0 including absorbed mass. Target max_t|E[X_t]−x0|<1e-10 in float64 regression tests.

Neutral long-time fixation: P_fix(infinity)=x0, P_loss(infinity)=1−x0. Run long enough that interior probability is below 1e-8.

Neutral heterozygosity: compare with 2*x0*(1−x0)*exp(−t/(2*Ne)), checking convergence across resolutions and allowing initial projection error.

For drift M=s*x*(1−x) and V=x*(1−x)/(2*Ne), compare long-time fixation against

\[
u(x_0)=\frac{1-e^{-4N_esx_0}}{1-e^{-4N_es}},\qquad s\ne0.
\]

Use explicit custom drift for this check; the dominance convention can differ. The implemented callable signature is `drift_override=lambda x,t: s*x*(1-x)`.

For neutral bidirectional mutation, derive the Beta stationary exponents from M and V rather than hard-coding unexplained constants. Compare projected interior mass in L1 and moments.

### 10.2 Independent references

Compare small grids with dense `scipy.linalg.expm` (tests only), individual-based Wright–Fisher Monte Carlo with statistical uncertainty, and refined spatial/time self-references. Fix and record Monte Carlo seeds, use multiple seeds, and judge against theoretical standard errors.

### 10.3 Convergence

For G=250,500,1000,2000 and a sequence of time steps, measure final

\[
\|p^{(G)}-R(p^{(2G)})\|_1,
\]

where R conservatively transfers fine-grid mass to the coarse grid. Record neutral convergence trends; permit first-order overall convergence with upwind advection. Verify consistently decreasing errors rather than relying only on a strict order fit.

## 11. Test specification

### 11.1 Unit tests

Test exact endpoints/uniformity/monotonicity/point validation; variance zero at endpoints and nonnegative inside; known selection/mutation drift; rejection of nonpositive Ne and negative mutation; delta mass and mean preservation; zero generator row sums, nonnegative off-diagonals, nonpositive diagonals, zero absorbing rows; statistics including endpoints; endpoint plus interior mass summing to one; NPZ array/metadata round trips; unknown-schema rejection.

### 11.2 Integration tests

Require API/CLI agreement for neutral absorption, expm versus sufficiently small implicit steps, correct bottleneck splitting, mutation re-entry, and strict increasing-time validation with duplicate rejection rather than silent sorting.

### 11.3 Property tests

Across randomly generated valid parameters, check unit total probability, no excessive negativity, nondecreasing absorbing endpoint probabilities, and neutral mean preservation at all saved times.

### 11.4 Regression tests

Save small reference outputs. Compare summary statistics with tight absolute/relative tolerances and full distributions with L1 distance. Report performance regressions in benchmarks without failing CI on an absolute runtime threshold across machines.

### 11.5 Acceptance

All unit, integration, convergence, and regression tests must pass. Require maximum pre-correction conservation error <1e-12 for neutral expm, <1e-10 for implicit cases, and no probability below −1e-12 in standard cases. Complete G=100000 without densification and pass tests natively on arm64.

## 12. Performance requirements

### 12.1 Complexity

Generator construction and each tridiagonal implicit step require O(G) time and workspace. Sparse exponential workload depends on the problem; each matrix-vector product is O(G), with O(G)–O(TG) storage depending on outputs. Full output costs O(TG) time and memory.

### 12.2 Targets

Record same-machine release benchmarks rather than fixed absolute runtime acceptance. Cover G=1000/10000/100000, neutral/strong-selection/mutation, expm/implicit Euler, and T=10/100/1000. Record peak RSS, operator construction, integration, and saving.

Avoid O(G²) storage, repeated same-shape work-array allocation, and retention of unrequested intermediate states. If T*G*8 bytes exceeds a configurable cap, suggest streaming or summary-only output; these may require a later release.

### 12.3 Profiling

Separately report coefficient evaluation, generator construction, factorization, evolution, diagnostics, and serialization, as well as wall time.

## 13. Apple Silicon optimization

### 13.1 Environment

Use native arm64 Python and dependencies; record platform.machine(). Permit Rosetta x86_64 with a startup warning and separate benchmark labeling. Expose BLAS/LAPACK configuration equivalent to numpy.show_config().

### 13.2 CPU baseline

One-dimensional tridiagonal/sparse operations have memory-transfer and sequential-dependency costs; GPU launch/transfer can outweigh benefits for one case. SciPy CPU is the official v1 backend. Optimize in order: avoid dense matrices; exploit tridiagonal solvers; reuse factors/buffers; vectorize coefficients and reduce copies; parallelize independent cases.

### 13.3 BLAS threads

Compare one thread against automatic threading. Excess BLAS threads may hurt one-dimensional sparse work. For process-based sweeps, limit each process to one BLAS thread to prevent oversubscription.

### 13.4 MPS / MLX

Future experimental batched GPU backends must process many independent parameter sets on a common grid, match CPU conservation/positivity/convergence, quantify float32 error and disclose unavailable float64, and show end-to-end gains including transfers. Never remove the CPU reference or validation path to accommodate GPU support.

### 13.5 Parallel sweeps

Distribute independent cases through ProcessPoolExecutor or equivalent instead of parallelizing inside one trajectory. Choose conservative worker defaults based on physical cores, memory, and BLAS threads; permit user limits. Save each case atomically to its own file and provide a resumable manifest.

## 14. Logging and reproducibility

Offer ERROR/WARNING/INFO/DEBUG. INFO should report normalized model summary, points/dx, method/time range, maximum Péclet and resolution warnings, steps/duration, maximum conservation error/minimum probability, and output destination. Never log full distributions. Attach normalized configuration and environment to research artifacts.

The deterministic PDE solve uses no random numbers. The original scope used randomness only for Monte Carlo validation, recording seed, generator, and repetitions. The later trajectory extension also records these sampling details separately from the deterministic solve.

## 15. Implementation phases

### Phase 1: minimal verifiable implementation

Uniform grid; neutral/selected models without mutation; absorption; CTMC generator; expm_multiply; Python API; conservation, neutral mean, and fixation tests.

### Phase 2: research usability

Bidirectional mutation and mutation boundaries; time-dependent Ne,s,u,v; backward Euler; YAML CLI; NPZ/CSV; plots; convergence tests; benchmarks; metadata.

### Phase 3: accuracy and large sweeps

Chang–Cooper; adaptive stepping; streaming; parallel/resumable sweeps; nonuniform grids. Clarification: basic step-doubling was implemented earlier to resolve §4.6's first-release behavior.

### Phase 4: inference and multidimensional foundations

Sensitivity/adjoint or differentiable backend; multi-population operator splitting; batched GPU; likelihood integration.

## 16. Future extensions

### 16.1 Nonuniform/adaptive grids

Refine near endpoints and initial frequencies. Carry cell volumes with probability mass and preserve mass and first moments as far as possible during regridding.

### 16.2 Migration and population structure

For external frequency x_m(t), add M_mig=m(t)*(x_m(t)−x). Multiple populations expand the state space as G^d; consider ADI, splitting, low-rank representations, or spectral methods.

### 16.3 Arbitrary demography

Add piecewise-linear schedules, exponential growth, and user demography. Use a common schedule interface exposing discontinuities to integrators.

### 16.4 Multiple alleles/loci

Multiallelic PDEs live on a simplex. Do not force the one-dimensional API and storage format onto them; share metadata, schedules, and diagnostics interfaces where appropriate.

### 16.5 Inference

Add observation likelihoods, sampling models, and gradients as separate modules. Keep the observation process separate from the forward solver.

### 16.6 Interoperability

Consider xarray/Zarr, Jupyter visualization, comparisons/conversions with dadi or moments, and C/C++/Rust tridiagonal kernels only if profiling demonstrates a need.

## 17. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Confusing V and D=V/2 | Factor-of-two time error | Public variance naming and analytical tests |
| Mixing endpoint mass into density | Wrong normalization/moments | Canonical mass representation; no endpoint density |
| Upwind diffusion under strong selection | Oversmoothing | Péclet warning, refinement, future Chang–Cooper |
| Coarse grid for new mutations | Excess initial loss | dx<=x0/2 warning; future nonuniform grids |
| Crank–Nicolson oscillation | Negative probability | Backward Euler default and strict checks |
| Coefficient discontinuities | Integration error | Always split at breakpoints |
| Reduced GPU precision | Inconsistent research results | Retain CPU float64 reference |
| Very large output arrays | Memory exhaustion | Preflight estimates; future streaming/summary-only modes |

## 18. Definition of done

The first research-oriented release is complete when Phase 1 and Phase 2 are implemented; API/CLI examples agree; absorbing loss/fixation are explicitly tracked; automated tests cover conservation, positivity, neutral means/fixation, and time/space convergence; float64 arm64 acceptance passes; full declarative configuration and environment can be recovered from results; benchmarks demonstrate an O(G) workspace path through G=100000; and README/API docstrings state generations, diploidy, and V=x*(1−x)/(2*Ne).

Callable code/closures cannot be restored from metadata; this limitation is explicit in the review. Saved output remains O(TG), distinct from integration workspace.

The central principle is to implement **probability mass, boundary inflow, and conservation correctly before optimizing the visual appearance of density plots**.
