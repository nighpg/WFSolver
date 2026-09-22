# Implementation validation

Date: 2026-09-22. Environment: arm64 / Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1.

## Original solver acceptance baseline

`python -m pytest -q`: **78 passed in 17.80 seconds**, before Web UI and trajectory extensions.

Coverage: generators, initial projection, dense exponential references, conservation/nonnegativity, neutral and selected fixation, mutation, time dependence, spatial/time convergence, multiple-seed Monte Carlo, persistence/CLI, and implicit integration over 100,000 intervals.

## Hardware benchmarks

All 9 basic cases and all 54 combinations of grid size, method, model, and output count completed.

- Maximum pre-correction conservation error, expm: 1.18794e-14 (acceptance <1e-12).
- Maximum pre-correction conservation error, implicit: 1.53211e-13 (acceptance <1e-10).
- Minimum probability: −1.05487e-31.
- Total negative-value correction count: 1777.

Implicit integration with 100,000 intervals, Ne=10,000, duration 1 generation, maximum dt=0.1, and 10 output times:

| Model | Solve time (s) | Peak RSS (MiB) | Maximum conservation error |
|---|---:|---:|---:|
| Neutral | 0.0334 | 126.5 | 5.66e-14 |
| Selection | 0.0309 | 127.8 | 1.09e-13 |
| Mutation | 0.0313 | 126.5 | 1.05e-13 |

The 54-case matrix uses 1,000/10,000/100,000 intervals, expm/implicit Euler, neutral/selection/mutation, 10/100/1000 output times, duration 0.001 generations, and maximum dt=0.0001. Maximum peak RSS was 1017.8 MiB. At 1000 outputs and 100,000 intervals, saved mass alone occupies approximately 763 MiB: retaining all outputs is O(TG).

Integration workspace is O(G). Peak RSS includes Python, libraries, and saved arrays, not just work buffers. These short-duration benchmarks do not guarantee long-duration performance.

Data: [basic cases](acceptance_arm64.json), [full matrix](matrix_arm64.json), [environment packages](requirements-arm64.txt). Commands are in [README](README.md).

## Examples and scope

The neutral, bottleneck, and mutation YAML examples were run through the CLI, generating NPZ and summary CSV in `results/`. The neutral PNG was visually checked.

This baseline validates generations, uniform grids, and float64. Scaled time, GPU, nonuniform grids, Chang–Cooper, streaming, and parallel sweeps are outside its scope. Later Web UI and trajectory checks are additional to these solver benchmarks.

## Web UI and trajectory extension

After the English UI/documentation and trajectory extension: `python -m pytest -q` reports **105 passed in 18.02 seconds**. Additional checks cover seeded NPZ/CSV and CLI output, temporal correlations and CTMC marginals, absorbing endpoints, scheduled mutation re-entry, first-saved-time initialization, invalid inputs/event budgets, Web sampling concurrency/cancellation/timeouts, and preservation of the last successful sample.

Browser verification on the local server confirmed English controls, simulation execution, trajectory plotting, reload restoration, time navigation, and the fixed density y-axis, with no captured console errors. For Ne=1000, 501 points, neutral absorption, x0=0.3, 51 outputs through generation 10000, sampling 32 paths with seed 42 generated 1,791,256 jumps in approximately 2.35 seconds. This is an illustrative measurement, not a performance guarantee. JavaScript syntax validation and offline wheel construction also passed.

## Demographic SFS extension

Both PDF-symmetric and standard-derived neutral expected SFS modes were added. The complete suite reports **133 passed in 19.75 seconds**. After adding environment metadata, the 26 SFS numerical/CLI checks were rerun and passed. Validation includes exact equilibrium spectra through n=2140, polynomial differentiation against the continuous neutral diffusion, independent dense affine matrix exponentials for population changes, linear-history tolerance convergence, hypergeometric downprojection consistency, folded midpoint handling, zero theta, rejection of unsupported settings, and Web job isolation/downloads/cancellation.

Browser checks confirmed both curves, folded and normalized displays, the independent time slider, restoration after reload, and a 2140-copy growth calculation, with no captured browser console errors. A three-output 2140-copy growth calculation took approximately 0.047 seconds in a local API check; this is illustrative, not an acceptance threshold. This uses exact neutral sample-moment closure with numerical BDF time integration. It does not validate selection or demographic inference, which are outside the SFS feature's scope. See `docs/sfs.md` for the explicit mirrored-source convention used to complete the PDF's symmetric demographic model.
