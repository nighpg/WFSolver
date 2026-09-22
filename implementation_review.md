# Wright–Fisher forward solver: implementation review

Scope: `wright_fisher_forward_solver_spec.md`. Review date: 2026-09-22.

This is the historical pre-implementation design review. At the time, the workspace contained only the specification; implementation, measurements, and acceptance tests had not yet been performed. The clarifications below were proposals, not changes to the original requirements. For implemented behavior, see the current README; for measured results, see `benchmarks/validation.md`.

## Assessment

Python, NumPy, SciPy, float64, and CPU are a viable baseline. Evolving nodal probability mass as a CTMC with explicit endpoint states is appropriate. Establish generator identities and analytical properties before adding time dependence, persistence, and CLI integration. Main risks concern boundary semantics, initial-density projection, callable contracts, and large exponential-action workloads.

## 1. Clarifications before implementation

### 1.1 Mutation endpoint probabilities (specification §§5, 7.3)

With absorption, `p_loss=p[0]` and `p_fix=p[-1]` are cumulative loss/fixation probabilities. With mutation, endpoints permit re-entry, so these are occupancy probabilities, not cumulative absorption or first-hit probabilities.

Provide general names `p_at_zero` and `p_at_one`. If compatibility aliases are retained, explain mutation semantics in API, CSV, and plots. Test one-way mutation: u=0,v>0 leaves 0 absorbing; u>0,v=0 leaves 1 absorbing. The bidirectional continuous stationary distribution has no endpoint atoms, whereas a finite CTMC may have positive stationary endpoint mass; these are different quantities.

### 1.2 Initial density and boundaries (specification §§4.7, 10.1)

The density-to-mass projection needs a definition. Pointwise evaluation cannot handle Beta endpoint singularities when a shape parameter is below one. Extend linear delta allocation by integrating the density against nodal piecewise-linear hat functions. Exact integration preserves total mass and mean. Use incomplete Beta functions for Beta densities. Define density-array shape and interpolation explicitly.

Even this projection allocates some near-boundary density to endpoints. Record and warn about resulting initial absorption error; it does not resolve sub-grid frequencies. For Ne=10000, a new mutation has x0=0.00005. At G=2000, linear allocation places 90% at zero initially. The recommendation dx<=x0/2 requires G>=40000, or at least 40001 points.

Validate integrated mass for arbitrary densities and reject invalid input by default. Distinguish roundoff in analytically normalized Beta distributions from explicit normalization of user inputs.

### 1.3 Time dependence and automatic stepping (specification §§4.5–4.6, 15)

The specification describes automatic stepping in §4.6 but puts it in Phase 3 in §15. Clarify first-release behavior:

- Include basic step-doubling. Fixed dt splits only as needed at output times and breakpoints.
- Accept the two-half-step solution, without Richardson extrapolation that may break positivity.
- Expose atol, rtol, min_dt, max_dt, max_steps, and a rejection limit.
- Arbitrary callables do not reveal their shortest variation timescale. Accept optional parameter_timescale and breakpoints; otherwise initialize from output spacing, without guaranteeing detection of rapid oscillations or jumps. Include h(t) in timescale considerations.
- Schedules are right-continuous; use the union of all breakpoints and retain the last value thereafter.
- Initial conditions refer to t=0 even when outputs begin later. Reject nonfinite, negative, unsorted, and duplicate output times.
- Add method="auto". Reject explicit expm for general time dependence; interval-wise exponentials for piecewise-constant models can be a separate future feature.

### 1.4 Model API and reproducibility (specification §§3.3, 7.2, 18)

Expose `drift_override(x,t)` to replace the complete drift, including selection and mutation. The original `lambda x` validation example does not match the proposed drift(x,t) signature.

Infer time homogeneity from scalar/schedule structure or an explicit guarantee, never a few sampled function values. Reject inward nonzero endpoint drift in absorbing mode instead of silently replacing it with absorption. Validate finite values, shape, and model constraints at every callable evaluation.

Callable names and source hashes cannot restore closures, external data, or environments. Guarantee complete configuration reconstruction only for built-in models and declarative schedules; record identifying information and external dependencies for callables.

If scaled time is introduced, define `M_scaled=2*reference_Ne*M_generation` and `V_scaled=2*reference_Ne*V_generation`, including units for time-dependent functions and breakpoints.

### 1.5 Probability corrections and diagnostics (specification §§7.4, 11.5)

The condition min(p)>=−1e-14 also includes entirely positive arrays. Renormalizing these routinely could hide conservation bugs. Check finiteness, total mass, and minimum **before correction**; fail outside tolerances; clip only actual tiny negatives; then record renormalization, negative mass, and L1 correction. Use pre-correction errors for acceptance.

Compute Péclet numbers at interior nodes. If D=0, use infinity when M is nonzero and zero when both vanish; avoid NaNs and endpoint 0/0 in maxima.

## 2. Numerical structure

Adopt the proposed modules, adding schedules.py and exceptions.py.

| Module | Responsibility |
|---|---|
| grid.py | Uniform grid, delta allocation, density projection, integer point validation |
| coefficients.py / schedules.py | Coefficients, validation, homogeneity, breakpoints |
| operator.py | One-dimensional lower/upper/diagonal rate arrays; CSC when needed |
| integrators.py | Exponential action, tridiagonal implicit integration, fixed/adaptive steps, interval splitting |
| diagnostics.py | Pre-correction conservation, negativity, Péclet numbers, correction history |
| solver.py | Validation, time management, method selection, output collection |
| result.py / io.py | Statistics, NPZ round trips, CSV, schema validation |
| cli.py / plotting.py | Declarative YAML, exit codes, saving, plotting |

Install NumPy/SciPy as standard dependencies; put Matplotlib in plot, PyYAML in cli, and pytest in dev extras. If core dependencies are optional, clearly document installation.

The transpose is the main tridiagonal indexing risk. For a_i=q[i,i−1], b_i=q[i,i+1], A=I−dt*Q.T:

- A[i,i] = 1+dt*(a_i+b_i)
- A[i,i−1] = −dt*b_(i−1)
- A[i,i+1] = −dt*a_(i+1)

First compare solve_banded with an independent reference. Reuse factorizations through separate LAPACK factor/solve calls. Key bounded caches by operator identity and dt; adaptive stepping must not cause unlimited cache growth. References: [SciPy solve_banded](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html), [SciPy dgttrf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgttrf.html).

## 3. Performance considerations

For neutrality, maximum exit rate is approximately G²/(8*Ne). At G=100000, Ne=10000 this is 125000 per generation. O(G) sparse operations do not guarantee few operations for long-time exponential action. This is an analytical workload estimate, not measured timing.

[expm_multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html) provides exponential action and equally spaced output; use it as the time-homogeneous reference method. A G=100000 acceptance case must specify Ne, duration, accuracy, output count, and integrator. Implement tridiagonal implicit integration early, verify temporal convergence, and never silently substitute methods.

Full output storage is O(TG). T=1000,G=100000 needs about 800 MB (763 MiB) for mass alone, plus workspace. Distinguish O(G) integration memory from O(TG) output. If streaming is deferred, estimate output storage before allocation and fail explicitly above a configurable cap.

## 4. Validation design

Generator identities can be checked independently of integration:

- Q @ ones = 0 for every model.
- Q @ x = 0 under neutral absorption.
- For H_i=2*x_i*(1−x_i), Q @ H = −H/(2*Ne) under neutral absorption.

Thus exact neutral CTMC evolution gives E[H(t)]=E[H(0)] exp(−t/(2*Ne)). Off-grid delta projection changes E[H(0)] slightly from 2*x0*(1−x0); separate initial projection error from integration error.

Zero-flux stationarity gives rho(x) proportional to `1/V(x) * exp(integral(2*M/V) dx)`, hence Beta(4*Ne*u,4*Ne*v) under neutral bidirectional mutation. Compare projected reference mass, total mass, endpoint discretization error, mean, and variance rather than density at interior points alone.

Additional checks:

- Dense exponential references and independently assembled tridiagonal systems on small grids.
- Time-varying neutral Ne: E[H(t)]=E[H(0)] exp(−integral dt/(2*Ne(t))) to test breakpoint splitting.
- Separate custom drift fixation tests from dominance selection: at h=1/2, drift is (s/2)*x*(1−x).
- Evaluate fixation formulas with expm1 and sign-dependent forms to avoid cancellation and overflow.
- Endpoint initial deltas, one-way mutation, unsorted times, one output time, and coincident final output/breakpoint.
- Conservative fine-to-coarse mass projection for L1 convergence; report endpoint error separately from mass projected near endpoints.
- Compare finite-population Monte Carlo in weak-selection regimes where diffusion is appropriate; distinguish statistical error from model approximation.

## 5. Recommended sequence

1. Clarify boundary mass, initial projection, units, and callable contracts; build package, grid, model, and exceptions.
2. Phase 1: absorbing CTMC, exponential action, statistics, generator identities, neutral moments/fixation, independent small-grid references.
3. Phase 2 numerics: mutation, schedules, tridiagonal implicit integration, breakpoint splitting, basic adaptivity, spatial/time convergence.
4. Phase 2 usability: YAML CLI, NPZ/CSV, plots, metadata, API/CLI agreement, invalid inputs and exit codes.
5. Acceptance: native arm64, pre-correction conservation, defined 100000-interval benchmarks, examples and mathematical conventions.

Defer Chang–Cooper, Crank–Nicolson, GPU, nonuniform grids, and parallel sweeps. Prioritize correct boundaries, conservation, and convergence.
