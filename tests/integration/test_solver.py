import numpy as np
import pytest
from scipy.linalg import expm

from wf_forward import *
from wf_forward.operator import build_generator


@pytest.mark.parametrize("times", [[0, .5, 1], [.5, 1], [1], [0], [0, .3, 1]])
def test_expm_matches_dense(times):
    model, grid = WrightFisherModel(Ne=100, selection=.01), UniformGrid(101)
    initial = DeltaInitialCondition(.304)
    result = ForwardSolver(model, grid, SolverConfig(output_times=times)).solve(initial)
    q = build_generator(model, grid).as_sparse().toarray()
    expected = np.array([expm(q.T*t) @ initial.to_mass(grid) for t in times])
    np.testing.assert_allclose(result.probability_mass, expected, atol=5e-15)
    assert result.diagnostics["max_probability_error"] < 1e-12


@pytest.mark.parametrize("method", ["expm", "implicit_euler"])
def test_neutral_moments_and_absorption(method):
    grid = UniformGrid(101)
    initial = DeltaInitialCondition(.304)
    times = np.arange(0, 11, 2)
    result = ForwardSolver(WrightFisherModel(Ne=100), grid,
                           SolverConfig(output_times=times, method=method, dt=.25)).solve(initial)
    np.testing.assert_allclose(result.mean(), .304, atol=1e-12)
    assert np.all(np.diff(result.p_loss) >= -1e-14)
    assert np.all(np.diff(result.p_fix) >= -1e-14)
    np.testing.assert_allclose(result.p_loss+result.p_fix+result.p_segregating, 1, atol=1e-12)
    H0 = initial.to_mass(grid) @ (2*grid.x*(1-grid.x))
    expected = H0*np.exp(-times/200) if method == "expm" else H0*(1+.25/200)**(-times/.25)
    np.testing.assert_allclose(result.heterozygosity(), expected, atol=2e-13)
    if method == "implicit_euler":
        assert result.metadata["integration"]["factorizations"] == 1
        assert result.metadata["integration"]["steps"] == 40


def test_neutral_long_time_fixation():
    result = ForwardSolver(WrightFisherModel(Ne=100), UniformGrid(101),
                           SolverConfig(output_times=[4000])).solve(DeltaInitialCondition(.3))
    assert result.p_segregating[-1] < 1e-8
    assert result.p_fix[-1] == pytest.approx(.3, abs=1e-8)
    assert result.p_loss[-1] == pytest.approx(.7, abs=1e-8)


@pytest.mark.parametrize("u,v,endpoint", [(.001, .002, 0), (.001, 0, 0), (0, .001, 1)])
def test_mutation_reentry_and_mean(u, v, endpoint):
    result = ForwardSolver(WrightFisherModel(Ne=100, mutation_forward=u, mutation_backward=v,
                                           boundary="mutation"), UniformGrid(201),
                           SolverConfig(output_times=[0, 1, 5])).solve(DeltaInitialCondition(endpoint))
    assert result.p_segregating[-1] > 0
    stationary_mean = u/(u+v)
    expected = stationary_mean+(endpoint-stationary_mean)*np.exp(-(u+v)*result.times)
    np.testing.assert_allclose(result.mean(), expected, atol=1e-13)
    assert result.metadata["endpoint_semantics"] == "occupancy"


def test_one_way_mutation_absorbing_endpoint():
    model = WrightFisherModel(Ne=100, mutation_forward=.001, boundary="mutation")
    result = ForwardSolver(model, UniformGrid(101), SolverConfig(output_times=[0, 10])).solve(DeltaInitialCondition(1))
    np.testing.assert_allclose(result.p_at_one, [1, 1], rtol=0, atol=1e-13)


def test_breakpoint_splitting_and_single_output():
    model = WrightFisherModel(Ne=PiecewiseConstant([0, .3, .8], [100, 20, 100]))
    result = ForwardSolver(model, UniformGrid(101),
                           SolverConfig(output_times=[1], dt=.5)).solve(DeltaInitialCondition(.3))
    expected = .42/((1+.3/200)*(1+.5/40)*(1+.2/200))
    assert result.heterozygosity()[0] == pytest.approx(expected, abs=1e-14)
    assert result.metadata["integration"]["steps"] == 3
    assert result.metadata["integration"]["operator_builds"] == 2


def test_adaptive_improves_with_tolerance():
    model, grid = WrightFisherModel(Ne=PiecewiseConstant([0, .3], [100, 20])), UniformGrid(101)
    initial = DeltaInitialCondition(.3)
    p = initial.to_mass(grid)
    q0 = build_generator(model, grid, 0).as_sparse().toarray()
    q1 = build_generator(model, grid, .5).as_sparse().toarray()
    expected = expm(q1.T*.7) @ expm(q0.T*.3) @ p
    errors = []
    for tolerance in (1e-3, 1e-5):
        result = ForwardSolver(model, grid, SolverConfig(output_times=[0, 1], atol=1e-10,
                                                       rtol=tolerance)).solve(initial)
        errors.append(np.abs(result.probability_mass[-1]-expected).sum())
        assert result.diagnostics["max_probability_error"] < 1e-10
    assert errors[1] < errors[0]/3


def test_adaptive_failure_is_explicit():
    model = WrightFisherModel(Ne=lambda t: 100)
    with pytest.raises(NumericalError):
        ForwardSolver(model, UniformGrid(101), SolverConfig(output_times=[0, 1],
                      min_dt=.1, atol=1e-15, rtol=1e-15)).solve(DeltaInitialCondition(.3))


def test_random_valid_models_preserve_probability():
    rng = np.random.default_rng(731)
    for _ in range(12):
        model = WrightFisherModel(Ne=rng.uniform(50, 500), selection=rng.uniform(-.1, .1),
                                 dominance=rng.uniform(0, 1))
        result = ForwardSolver(model, UniformGrid(101),
                               SolverConfig(output_times=[0, .5, 2])).solve(DeltaInitialCondition(rng.uniform(.1, .9)))
        assert result.probability_mass.min() >= 0
        assert result.diagnostics["max_probability_error"] < 1e-12
        assert np.min(np.diff(result.p_loss)) >= -1e-14
        assert np.min(np.diff(result.p_fix)) >= -1e-14


def test_neutral_monte_carlo_independent_check():
    # Finite diploid Wright–Fisher vs diffusion, allowing finite-Ne bias in H.
    result = ForwardSolver(WrightFisherModel(Ne=50), UniformGrid(201),
                           SolverConfig(output_times=[10])).solve(DeltaInitialCondition(.3))
    means, hs = [], []
    for seed in (123, 456, 789):
        rng = np.random.default_rng(seed)
        x = np.full(20000, .3)
        for _ in range(10):
            x = rng.binomial(100, x)/100
        means.append(x.mean())
        hs.append((2*x*(1-x)).mean())
    finite_h = .42*(1-1/100)**10
    se_mean = np.sqrt(float(result.variance()[0])/60000)
    assert abs(np.mean(means)-result.mean()[0]) < 5*se_mean
    assert abs(np.mean(hs)-finite_h) < .003


def test_selection_forward_fixation_matches_discrete_hitting_probability():
    s, Ne, G = .01, 100, 100
    model = WrightFisherModel(Ne=Ne, drift_override=lambda x, t: s*x*(1-x), time_homogeneous=True)
    result = ForwardSolver(model, UniformGrid(G+1), SolverConfig(output_times=[4000])).solve(DeltaInitialCondition(.3))
    log_ratio = -np.log1p(4*Ne*s/G)
    discrete = np.expm1(log_ratio*30)/np.expm1(log_ratio*G)
    assert result.p_segregating[-1] < 1e-8
    assert result.p_fix[-1] == pytest.approx(discrete, abs=1e-8)


def test_mutation_forward_approaches_stationary_mass():
    model = WrightFisherModel(Ne=100, mutation_forward=.01, mutation_backward=.02, boundary="mutation")
    grid = UniformGrid(101)
    with pytest.warns(ResolutionWarning):
        result = ForwardSolver(model, grid, SolverConfig(output_times=[500])).solve(DeltaInitialCondition(0))
    q = build_generator(model, grid)
    logp = np.concatenate(([0], np.cumsum(np.log(q.up[:-1])-np.log(q.down[1:]))))
    stationary = np.exp(logp-logp.max())
    stationary /= stationary.sum()
    assert np.abs(result.probability_mass[-1]-stationary).sum() < 2e-6
