# Demographic expected site frequency spectra

The Web UI, Python API, and CLI calculate **neutral infinite-sites expected site counts** under a time-dependent effective population size. This calculation has its own history and ancestral equilibrium; it does not project a single-locus delta simulation or use the single-locus selection/mutation/grid controls.

## Two explicit conventions

At constant reference population size N0, with i=1,...,n−1:

| Mode | Initial population-frequency intensity | Expected sample spectrum |
|---|---|---|
| Standard derived | theta/q | theta/i |
| PDF symmetric | theta/[q(1−q)] | theta*(1/i+1/(n−i)) |

The sampling equation is `g_i = integral BinomialPMF(i;n,q) * intensity(q) dq`, or equivalently theta times the integral of the per-unit-theta intensity. These singular intensities are not normalized probability densities. The outputs are expected numbers of polymorphic sites; their sum need not be one. No random sampling is required to calculate expectations.

**The symmetric mode is the sum, not the average, of the derived spectrum and its reflection.** At the same theta its total is twice the derived total. This reproduces the chosen symmetric equilibrium expression in the supplied supplementary PDF. For demographic evolution we explicitly extend it using mirrored initial spectra and equal continuous mutation sources at the two boundaries. The two-page excerpt does not uniquely specify those sources, so this is a documented completion of its model, not a claim to reproduce every detail of the original authors' implementation or fitted population history. This neutral extension satisfies `symmetric_i(t) = derived_i(t) + derived_(n−i)(t)` at every saved time.

Theta is the **reference equilibrium amplitude**. The program does not silently convert per-base mutation rates to theta. The supplied PDF writes theta=2*N0*U; a conventional one-sided diploid infinite-sites model instead gives theta=4*N0*mu*L when mu is a per-copy, per-base rate and L is sequence length. Input the intended amplitude directly; do not treat these conventions as interchangeable without specifying U. Both outputs are multiplied by theta exactly once.

## Population history and time

Enter absolute diploid effective sizes at increasing forward times in generations. Time 0 is the start of the modeled history (the past), not the present. The population is at neutral equilibrium at the first size before time 0. If that size differs from N0, the ancestral derived spectrum is `theta*(Ne_initial/N0)/i`.

The UI defaults to piecewise-linear interpolation, as described in the PDF; piecewise-constant interpolation is also available. Linear histories connect knot sizes; constant histories are right-continuous. The final size continues through the end time. Every knot must occur by the final requested output. The corresponding PDF demographic parameters are `lambda_k = Ne_k/N0`. Scaled time used internally is `tau = generations/(2*N0)`; the UI and exports always use generations.

Presets:

- Constant: the reference size throughout.
- Bottleneck: size falls to 10% at 40% of the duration and returns to the reference size at 60%.
- Recent growth: reference size until 80% of the duration, then increases to ten times that size at the end.

Preset knots follow the currently entered reference size and duration when the preset is selected. Editing those values later does not silently rewrite the history textarea.

## Numerical method

For the neutral diffusion, integrate the forward equation against the Bernstein/binomial sampling polynomials. Their degree-n moment equations close exactly. Consequently the sampled spectrum can be evolved directly, without approximating a singular population density on a uniform frequency grid and then applying inaccurate endpoint quadrature.

For per-unit-theta derived counts y_i, i=1,...,n−1, rho=Ne/N0, and tau=t/(2*N0):

\[
\frac{dy_i}{d\tau}=\frac{1}{2\rho(\tau)}\left[
(i-1)(n-i+1)y_{i-1}-2i(n-i)y_i+(i+1)(n-i-1)y_{i+1}\right]
+\frac n2\mathbf1_{i=1}.
\]

Missing boundary terms are zero. The initial values are `rho(0)/i`. For the symmetric convention, reflect the solution and add it; this is equivalent to adding an equal source at i=n−1. For n=2 both sources reach the same polymorphic class and correctly add.

The source is constant in generation time for constant mutation rate: n/(4*N0) per unit theta. Changes in Ne affect drift, not this sample-level mutation source. Two auxiliary states accumulate outflow from the sample's polymorphic classes, allowing a source/outflow balance check. They are **not population fixation or loss probabilities**, and are not the monomorphic components of a normalized single-locus distribution.

We integrate the sparse tridiagonal linear system using SciPy BDF with an analytic sparse Jacobian, restarting at history knots. Defaults are rtol=1e-7 and atol=1e-10 on per-unit-theta states. Numerical time error remains; the exact closure removes the frequency-grid discretization error for this neutral calculation. Tolerances are configurable through JSON/API. Nonfinite values or excessive negativity fail; tiny negative saved values are clipped and reported. No mass renormalization is performed. Diagnostics report mutation-input/outflow balance, minimum before correction, clipped mass, function evaluations, and factorizations.

Background on sample-moment methods: [Jouganous et al. (2017)](https://academic.oup.com/genetics/article/206/3/1549/6064248). Integrator: [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). The source and equilibrium conventions above are explicitly defined by this implementation.

## Web UI

Open the local server and scroll to **Expected site frequency spectrum**.

1. Enter sampled allele copies n (twice the number of diploid individuals), theta, reference N0, and end time.
2. Select a preset or enter `generation: effective population size` rows beginning at generation 0.
3. Select **Calculate SFS**. The UI saves 51 output times. Calculation can be cancelled.
4. Switch **Spectrum convention** between PDF symmetric and Standard derived without recalculating. Use **Compare both conventions** to overlay them.
5. Select unfolded or folded counts, expected sites or polymorphic proportions, and a linear or log y-axis. The dashed ancestral reference changes with the selected convention and folding.
6. Use the independent SFS time slider to inspect demographic change. Download CSV, NPZ, or the exact configuration.

Folding uses g_i+g_(n−i) for i<n/2 and only g_(n/2) at the midpoint for even n. Folding does not change a spectrum's total. Normalized proportions divide by the selected spectrum's polymorphic total. At theta=0 there are no expected sites; the UI explicitly reports that proportions are undefined and draws zero values (no log-scale curve).

The expressions shown in the convention selector are the constant-reference equilibrium formulas, not formulas asserted to hold under arbitrary demography. Neutrality, ancestral equilibrium, mutation supply, and history determine the evolving shape.

Inputs and plots remain separate from the single-locus results. Changed settings mark a previous SFS as stale. A failed or cancelled replacement retains the displayed completed result and download links while that job remains stored. Reload reconnects to the latest SFS job when it is complete or running. A server restart clears temporary jobs. The server retains the latest three simulation jobs and three SFS jobs independently; one calculation of any kind can run at a time.

## Python and CLI

```python
from wf_forward import calculate_sfs

result = calculate_sfs({
    'n': 100,
    'theta': 1,
    'reference_Ne': 10000,
    'history': {
        'times': [0, 8000, 10000],
        'sizes': [10000, 10000, 100000],
        'interpolation': 'linear',
    },
    'output_times': [0, 8000, 10000],
})
print(result.derived[-1])
print(result.symmetric[-1])
print(result.spectrum('derived', folded=True, normalized=True)[-1])
result.save_npz('results/sfs.npz')
result.to_csv('results/sfs.csv')
```

```bash
wf-forward sfs examples/sfs_growth.json --output results/sfs_growth.npz \
  --csv results/sfs_growth.csv
```

The CLI takes JSON. Use `examples/sfs_constant.json`, `examples/sfs_bottleneck.json`, or `examples/sfs_growth.json`. Arrays `derived` and `symmetric` have shape (saved_times,n−1). NPZ contains these arrays, times, and metadata_json and can be loaded with `numpy.load(..., allow_pickle=False)`.

CSV exports **both unfolded expected spectra** at all saved times with columns:

```text
time,derived_allele_count,n,derived_expected_sites,symmetric_expected_sites
```

Folding/normalizing in the UI does not alter the exported canonical counts. Both formats retain the full numerical counts; NPZ metadata includes the effective configuration and conventions.

## Limits and checks

This version supports neutrality, ancestral equilibrium, linear/constant size histories, 2–5000 sampled copies, and 1–201 output times/history knots. The Web worker has a 120-second limit. Selection, inference of population history from observed SFS, and non-equilibrium ancestral input are not implemented for the SFS calculation; unknown JSON fields are rejected. Existing single-locus selection remains available separately.

Tests cover both equilibrium formulas (including n=2140), polynomial identities against the continuous diffusion operator, independent dense exponential references for size changes, linear-history tolerance convergence, hypergeometric projection consistency across sample sizes, folding without midpoint double-counting, zero theta, time/reference scaling, input rejection, CLI round trips, and Web isolation/cancellation/downloads. Expected storage is O(Tn) and the operator is O(n).
