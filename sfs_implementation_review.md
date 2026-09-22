# SFS calculation and visualization: feasibility review

Reviewed 2026-09-22. This records the pre-implementation design review. The subsequent neutral implementation supports both symmetric and derived spectra through exact sample-moment closure; see docs/sfs.md for the implemented model, resolved conventions, and limitations.

## Source and interpretation

The supplied PDF, `41467_2015_BFncomms9018_MOESM937_ESM.pdf`, contains printed pages 22–23 of the supplementary methods for *Rare variant discovery by deep whole-genome sequencing of 1,070 Japanese individuals* (Nature Communications, 2015). Both pages were extracted and visually inspected. The relevant heading is **Demographic inference**.

Its sampling equation is

\[
g(i,n;\lambda)=\theta\int_0^1\binom ni q^i(1-q)^{n-i}f(q,T;\lambda)\,dq.
\]

Here n denotes the number of sampled allele copies, i the derived allele count, lambda the vector of relative population sizes at demographic knots, and theta the overall mutation scale. The text defines theta=2*N0*U, with U the mutation rate for the set of sites under study. Time T and the demographic knots are also needed; lambda alone does not specify the model. The PDF represents population size by connected linear segments and uses 12 intervals in its application.

The preceding variant-discovery section instead uses n for individuals and 2n chromosomes. The UI must explicitly label its sampling input **Sampled allele copies (n)** to avoid this notational switch.

The displayed g values are expected site counts, not probabilities required to sum to one. An optional relative spectrum is g_i/sum(g_1,...,g_(n−1)), conditional on being polymorphic in the sample.

## What can be added directly to the existing solver

The current `SimulationResult` stores normalized single-locus mass p_j(T) over grid nodes x_j, including loss and fixation. Define

\[
B_{ij}=\Pr(K=i\mid q=x_j)=\binom ni x_j^i(1-x_j)^{n-i},
\quad P(K=i\mid T)=\sum_j B_{ij}p_j(T).
\]

This is the distribution of the derived allele count when sampling a locus with the stored frequency distribution. The matrix multiplication is exact for the discrete measure represented by the result, apart from floating-point evaluation. No random trajectory simulation is necessary. Do not multiply nodal masses by dx again.

Include i=0,...,n when checking probability conservation. For a polymorphic-only view, display i=1,...,n−1 and normalize only if explicitly requested. Even an interior population frequency may produce i=0 or n in a finite sample; these classes are not determined solely by the population endpoint masses. If no polymorphic mass remains, show an empty conditional distribution rather than dividing by zero.

This operation can be implemented as result post-processing, without changing the numerical solver. It is useful, but must be labeled **Sample count probability** rather than passed off as a demographic infinite-sites SFS. Multiplying by an arbitrary theta does not establish the mutation history required by that SFS.

## Additional work for a demographic expected-site-count SFS

A population-wide SFS requires a density/intensity of sites, including its ancestral initial spectrum and mutation input through time. The current `ForwardSolver` validates and conserves total mass one. Its finite-site bidirectional mutation drift is not equivalent to introducing mutations at new sites under an infinite-sites model.

A suitable extension would evolve nonnegative expected site masses w rather than unit-total probabilities:

\[
\dot w=Q_{\rm interior}(t)^\mathsf T w+b(t),
\qquad g_i(T)=\sum_j B_{ij}w_j(T).
\]

Here b represents calibrated new-mutation input, and the interior operator permits loss/fixation outflow. Alternatively evolve a per-unit-theta spectrum and multiply by theta at the end; do not include theta twice. Source calibration, endpoint treatment, initial equilibrium, and theta convention must be fixed together. Expected polymorphic-site mass need not be conserved; diagnostics should check positivity and the balance of mutation input and boundary outflow instead.

Reusable components include validated model coefficients, sparse/tridiagonal operator structure, linear solvers, output-time handling, and Web plotting infrastructure. Required additions include an SFS-specific state/result, source-aware integration and diagnostics, equilibrium initialization, sampling projection, and a declarative piecewise-linear demographic schedule. A callable can express a linear schedule in the Python API today, but is not reconstructible through the current JSON/Web configuration.

For large n, add endpoint-aware quadrature or regularized-density integration and assess whether a nonuniform solver grid is required. The paper explicitly used an uneven grid; matching its numerical method is broader than adding a bar chart to the current uniform-grid solver.

## Ambiguities in the PDF that must not be silently resolved

1. **Equilibrium spectrum and orientation.** Page 22 states F(q) proportional to 1/[q(1−q)]. This is symmetric. It does not by itself specify a polarized derived-frequency spectrum. The standard neutral infinite-sites, derived-oriented spectrum uses a 1/q shape under a one-sided mutation-origin convention. The symmetric expression can reflect symmetrization, but the excerpt does not explain that choice. Neither singular expression is a normalized density over the whole closed interval.

   Direct integration gives, for 1<=i<=n−1:

   \[
   f(q)=C/q\Rightarrow g_i=\theta C/i;
   \qquad f(q)=C/[q(1-q)]\Rightarrow
   g_i=\theta C\,n/[i(n-i)].
   \]

   Thus this affects the shape of the unfolded spectrum, not only its scale. These identities follow from the Beta integral and provide useful independent tests. Under symmetrization, folding must not inadvertently count the same sites twice.

2. **Mutation scale and missing boundary/initial conditions.** The excerpt defines theta=2*N0*U but does not fully define the boundary mutation injection, initial f, or the relationship between U and a per-copy mutation rate times sequence length. Standard diploid conventions often use theta=4*Nref*mu*L. Keep the paper's theta as an explicit scale until its convention is confirmed; do not silently replace it. For comparison, dadi's official documentation defines theta0=4*Nref*u and implements the neutral density proportional to 1/q: https://dadi.readthedocs.io/en/latest/autoapi/dadi/PhiManip/ . The supplied two-page excerpt alone does not establish a unique full reproduction of the demographic calculation.

3. **Apparent notation errors.** Spatial derivatives in equations (2) and (3) are printed with time variables in their denominators, though they should be frequency derivatives for the stated diffusion. “Units of 2N0 generations” implies scaled time T=t_generation/(2N0), whereas the printed relation gives the opposite factor. The folded likelihood also has inconsistent g argument order and product limits relative to X=(x1,...,x_(n/2)). Implement a consistent mathematical definition and document the corrections rather than translating these expressions literally.

4. **Selection convention.** The PDF uses M=s_paper*q*(1−q). The current model at h=0.5 gives M=(s_code/2)*q*(1−q). Use s_code=2*s_paper when matching that genic drift, or expose the drift convention explicitly. A neutral first version avoids this ambiguity but should not hide it for future selection support.

## Proposed Web UI

Add an English **Site frequency spectrum** panel with:

- Sampled allele copies n and selection of a saved time.
- Unfolded derived count versus folded minor count.
- Expected sites, per-unit-theta spectrum, or normalized polymorphic proportions for the SFS model; a separately labeled probability mode for existing single-locus results.
- Theta input when expected counts are available; no theta requirement for relative shape.
- Bar/line view, linear/log y-axis, singleton summary, and CSV/NPZ export with conventions recorded.
- Optional time playback and overlay of the constant-size neutral reference.

For folded counts, use h_i=g_i+g_(n−i) when i<n/2; if n is even, h_(n/2)=g_(n/2), without doubling. The default SFS plot excludes i=0,n. No likelihood fitting is needed to fulfill a calculation/visualization feature; estimating lambda from observed SFS would be a separate extension.

## Numerical feasibility checks performed

A scratch calculation projected the saved 501-point, 51-time neutral result through a binomial kernel:

| Sample n | Observed wall time | Maximum total-probability error | Maximum E[K]/n versus E[X] error |
|---:|---:|---:|---:|
| 100 | ~0.004 s | 1.03e-11 | 1.11e-16 |
| 2140 | ~0.062 s | 1.03e-11 | 1.33e-15 |

The total-probability residual matches the original forward result; projection did not introduce a material new residual. These are local illustrative timings, not performance guarantees. Use stable binomial PMFs/logarithms and block processing for large n or grids instead of allocating every time×count×grid combination.

A separate diagnostic deliberately used a naive interior-node quadrature of the analytic f=1/q spectrum. For n=2140, the singleton expectation per unit scale is exactly 1; G=500,2000,10000,40000 returned about 0.0599,0.5586,0.8968,0.9735 respectively. These values are not errors measured in a completed SFS implementation. They demonstrate why omitting the singular boundary region and merely increasing a uniform grid can badly underestimate rare variants. Boundary-aware integration and grid convergence are essential; a fast projection alone is not an accuracy validation for demographic SFS.

## Validation plan

Check the two Beta-integral identities above under explicitly distinct conventions; agreement with a Binomial(n,q0) for a grid-aligned point mass; conservation and E[K]=n*E[X] for normalized single-locus results; zero-error folding totals and an undoubled central class; endpoint and no-polymorphism cases; source/outflow accounting; stationary neutral SFS; time/grid convergence especially for singleton/doubleton classes; and demographic examples against an independent implementation using matching mutation/time/selection conventions.

## Decisions requested

- Is the main target the paper-style expected-site-count SFS, the current single-locus sample-count probabilities, or both as distinct modes?
- Should derived orientation follow the standard one-sided infinite-sites convention, or should the PDF's symmetric spectrum be reproduced? Original code or additional boundary/initial-condition details would help resolve exact paper reproduction.
- Are there desired n, population-history, and theta values? Without a numerical specification, begin with neutral constant size and compare bottleneck/growth, making n and theta editable. Demographic likelihood fitting remains outside this proposed first step.
