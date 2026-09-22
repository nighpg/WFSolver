import numpy as np
import pytest

from wf_forward import *
from wf_forward.diagnostics import ProbabilityDiagnostics
from wf_forward.integrators import ImplicitStepper, OperatorProvider
from wf_forward.operator import build_generator


@pytest.mark.parametrize("points", [True, 100, 100.5, float("inf"), "101"])
def test_invalid_grid(points):
    with pytest.raises(ConfigurationError):
        UniformGrid(points)


def test_grid_and_delta():
    grid = UniformGrid(101)
    assert grid.x[0] == 0 and grid.x[-1] == 1
    assert not grid.x.flags.writeable
    for x0 in (0, .00005, .304, .9, 1):
        p = DeltaInitialCondition(x0).to_mass(grid)
        assert p.sum() == pytest.approx(1, abs=1e-15)
        assert p @ grid.x == pytest.approx(x0, abs=1e-15)


@pytest.mark.parametrize("kwargs", [{"Ne": 0}, {"Ne": float("nan")}, {"Ne": True},
    {"mutation_forward": -1}, {"mutation_forward": .01}, {"time_unit": "scaled"},
    {"Ne": PiecewiseConstant([0, 2], [10, -1])}, {"ploidy": 1}])
def test_invalid_model(kwargs):
    with pytest.raises(ConfigurationError):
        WrightFisherModel(**kwargs)


@pytest.mark.parametrize("times", [[], [-1, 0], [1, 0], [0, 0], [float("nan")], [True]])
def test_invalid_times(times):
    with pytest.raises(ConfigurationError):
        SolverConfig(output_times=times)


def test_schedule_and_runtime_validation():
    schedule = PiecewiseConstant([0, 2, 5], [10, 2, 20])
    assert [schedule(t) for t in (0, 1.99, 2, 4, 5, 100)] == [10, 10, 2, 2, 20, 20]
    with pytest.raises(ConfigurationError):
        PiecewiseConstant([1], [10])
    with pytest.raises(ConfigurationError):
        WrightFisherModel(Ne=schedule, time_homogeneous=True)
    model = WrightFisherModel(Ne=lambda t: 100 if t < 1 else -1)
    with pytest.raises(ConfigurationError):
        model.variance(np.array([.5]), 1)


@pytest.mark.parametrize("alpha,beta", [(1, 1), (2, 5), (.2, .3), (.1, 10), (10, .1)])
def test_beta_projection_mass_and_mean(alpha, beta):
    grid = UniformGrid(501)
    with pytest.warns(ResolutionWarning):
        p = BetaInitialCondition(alpha, beta).to_mass(grid)
    assert p.sum() == pytest.approx(1, abs=1e-12)
    assert p @ grid.x == pytest.approx(alpha/(alpha+beta), abs=1e-12)
    assert p.min() >= 0


def test_density_projection_and_normalization():
    grid = UniformGrid(101)
    with pytest.warns(ResolutionWarning):
        p = DensityInitialCondition(2*grid.x).to_mass(grid)
    with pytest.warns(ResolutionWarning):
        q = DensityInitialCondition(lambda x: 2*x).to_mass(grid)
    np.testing.assert_allclose(p, q, atol=1e-15)
    assert p @ grid.x == pytest.approx(2/3, abs=1e-13)
    with pytest.raises(ConfigurationError):
        DensityInitialCondition(np.ones(101)*2).to_mass(grid)
    with pytest.warns(ResolutionWarning):
        p = DensityInitialCondition(np.ones(101)*2, normalize=True).to_mass(grid)
    assert p.sum() == pytest.approx(1)
    with pytest.raises(ConfigurationError):
        ArrayInitialCondition(np.ones(101)).to_mass(grid)
    with pytest.raises(ConfigurationError):
        ArrayInitialCondition(np.zeros(101), normalize=True).to_mass(grid)


def test_coefficients_and_generator_invariants():
    grid = UniformGrid(101)
    model = WrightFisherModel(Ne=100, selection=.02, dominance=.3)
    x = grid.x
    np.testing.assert_allclose(model.drift(x), .02*x*(1-x)*(.3+.4*x))
    Q = build_generator(model, grid).as_sparse()
    assert Q.nnz <= 3*grid.points
    np.testing.assert_allclose(Q @ np.ones(grid.points), 0, atol=5e-15)
    assert np.all(Q.diagonal() <= 0)
    assert np.all(Q.diagonal(1) >= 0) and np.all(Q.diagonal(-1) >= 0)
    assert Q.getrow(0).nnz == 0 and Q.getrow(-1).nnz == 0
    neutral = build_generator(WrightFisherModel(Ne=100), grid).as_sparse()
    np.testing.assert_allclose(neutral @ x, 0, atol=4e-15)
    H = 2*x*(1-x)
    np.testing.assert_allclose(neutral @ H, -H/200, atol=4e-15)


def test_custom_drift_boundary_and_shape_validation():
    grid = UniformGrid(101)
    for drift in (lambda x, t: np.ones_like(x), lambda x, t: -np.ones_like(x),
                  lambda x, t: np.full_like(x, np.nan), lambda x, t: 0):
        with pytest.raises(ConfigurationError):
            build_generator(WrightFisherModel(drift_override=drift), grid)
    model = WrightFisherModel(drift_override=lambda x, t: .1*x*(1-x), time_homogeneous=True)
    assert model.is_time_homogeneous()
    assert build_generator(model, grid).max_peclet > 2


def test_tridiagonal_orientation_and_factor_reuse():
    grid = UniformGrid(101)
    model = WrightFisherModel(Ne=100, selection=.03, mutation_forward=.001,
                             mutation_backward=.002, boundary="mutation")
    provider = OperatorProvider(model, grid)
    stepper = ImplicitStepper(provider)
    p = DeltaInitialCondition(.31).to_mass(grid)
    q = build_generator(model, grid).as_sparse().toarray()
    expected = np.linalg.solve(np.eye(101)-.5*q.T, p)
    actual = stepper.advance(p, 0, .5)
    np.testing.assert_allclose(actual, expected, atol=2e-16)
    stepper.advance(actual, .5, .5)
    assert stepper.factorizations == 1


def test_diagnostics_before_correction():
    config = SolverConfig()
    diagnostic = ProbabilityDiagnostics()
    p = np.array([-.5e-14, .3, .7+.5e-14])
    diagnostic.check(p, config)
    assert p.min() == 0 and p.sum() == pytest.approx(1)
    assert diagnostic.correction_count == 1
    assert diagnostic.minimum_probability == -.5e-14
    positive = np.array([.5, .5+5e-11])
    unchanged = positive.copy()
    diagnostic.check(positive, config)
    np.testing.assert_array_equal(positive, unchanged)
    assert diagnostic.max_probability_error > 4e-11
    with pytest.raises(NumericalError):
        diagnostic.check(np.array([-.5e-14, .2, .7]), config)
    with pytest.raises(NumericalError):
        diagnostic.check(np.array([-1e-12, 1+1e-12]), config)


def test_unknown_method_and_memory_limit():
    with pytest.raises(ConfigurationError):
        SolverConfig(method="crank_nicolson")
    with pytest.raises(ConfigurationError):
        ForwardSolver(WrightFisherModel(Ne=lambda t: 100), config=SolverConfig(method="expm"))
    with pytest.raises(ConfigurationError):
        ForwardSolver(WrightFisherModel(), UniformGrid(101),
                      SolverConfig(output_times=[0, 1], max_output_bytes=100)).solve(DeltaInitialCondition(.3))
