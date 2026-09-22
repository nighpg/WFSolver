import warnings

import numpy as np
import pytest
from scipy.linalg import solve_banded

from wf_forward import *
from wf_forward.operator import build_generator


pytestmark = pytest.mark.convergence


def restrict_mass(fine):
    # Fine nodes at odd indices split between the neighboring coarse nodes.
    coarse = fine[::2].copy()
    coarse[:-1] += .5*fine[1::2]
    coarse[1:] += .5*fine[1::2]
    return coarse


@pytest.mark.parametrize("selection", [0, .02])
def test_spatial_distribution_convergence(selection):
    distributions = []
    for G in (250, 500, 1000, 2000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResolutionWarning)
            result = ForwardSolver(WrightFisherModel(Ne=1000, selection=selection), UniformGrid(G+1),
                                   SolverConfig(output_times=[10])).solve(BetaInitialCondition(3, 4))
        distributions.append(result.probability_mass[0])
    errors = [np.abs(coarse-restrict_mass(fine)).sum()
              for coarse, fine in zip(distributions, distributions[1:])]
    assert errors[1] < errors[0] and errors[2] < errors[1], errors


def test_temporal_first_order_convergence():
    model, grid = WrightFisherModel(Ne=100), UniformGrid(101)
    initial = DeltaInitialCondition(.3)
    reference = ForwardSolver(model, grid, SolverConfig(output_times=[1])).solve(initial).probability_mass[-1]
    errors = []
    for dt in (.2, .1, .05, .025):
        result = ForwardSolver(model, grid, SolverConfig(method="implicit_euler", output_times=[1], dt=dt)).solve(initial)
        errors.append(np.abs(result.probability_mass[-1]-reference).sum())
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-2]/errors[-1] > 1.8


@pytest.mark.parametrize("alpha,beta", [(2, 3), (.4, .7)])
def test_mutation_stationary_beta_convergence(alpha, beta):
    errors = []
    for G in (250, 500, 1000, 2000):
        grid = UniformGrid(G+1)
        model = WrightFisherModel(Ne=100, mutation_forward=alpha/400,
                                 mutation_backward=beta/400, boundary="mutation")
        q = build_generator(model, grid)
        logp = np.concatenate(([0], np.cumsum(np.log(q.up[:-1])-np.log(q.down[1:]))))
        p = np.exp(logp-logp.max())
        p /= p.sum()
        np.testing.assert_allclose(q.as_sparse().T @ p, 0, atol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResolutionWarning)
            reference = BetaInitialCondition(alpha, beta).to_mass(grid)
        errors.append(np.abs(p-reference).sum())
        assert p @ grid.x == pytest.approx(alpha/(alpha+beta), abs=1e-12)
    assert all(a > b for a, b in zip(errors, errors[1:])), errors


@pytest.mark.parametrize("s", [-.01, .01])
def test_selection_fixation_convergence(s):
    Ne, x0 = 100, .3
    # The reference formula uses M=s*x*(1-x), not h=.5 with the same s.
    exact = np.expm1(-4*Ne*s*x0)/np.expm1(-4*Ne*s)
    errors = []
    for G in (250, 500, 1000, 2000):
        grid = UniformGrid(G+1)
        model = WrightFisherModel(Ne=Ne, drift_override=lambda x, t: s*x*(1-x), time_homogeneous=True)
        q = build_generator(model, grid)
        # Independent backward hitting-probability equation Q_II*u=-Q_I,G.
        band = np.zeros((3, G-1))
        band[0, 1:] = q.up[1:-2]
        band[1] = q.diagonal[1:-1]
        band[2, :-1] = q.down[2:-1]
        rhs = np.zeros(G-1)
        rhs[-1] = -q.up[-2]
        fixation = np.concatenate(([0], solve_banded((1, 1), band, rhs), [1]))
        errors.append(abs(fixation[int(round(x0*G))]-exact))
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
