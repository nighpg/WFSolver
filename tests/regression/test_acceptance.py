import numpy as np
import pytest
import json
from pathlib import Path

from wf_forward import *


@pytest.mark.benchmark
def test_large_grid_without_dense_conversion(monkeypatch):
    import scipy.sparse

    def forbidden(*args, **kwargs):
        raise AssertionError("dense sparse-matrix conversion is forbidden")
    monkeypatch.setattr(scipy.sparse.csc_matrix, "toarray", forbidden)
    monkeypatch.setattr(scipy.sparse.csr_matrix, "toarray", forbidden)
    result = ForwardSolver(WrightFisherModel(Ne=10000), UniformGrid(100001),
                           SolverConfig(method="implicit_euler", dt=.1, output_times=[0, 1])).solve(DeltaInitialCondition(.3))
    assert result.probability_mass.shape == (2, 100001)
    assert result.diagnostics["max_probability_error"] < 1e-10
    assert result.probability_mass.min() >= 0
    np.testing.assert_allclose(result.mean(), .3, atol=1e-10)
    assert result.metadata["integration"]["factorizations"] == 1


def test_exact_neutral_implicit_regression():
    # Analytically specified moments, independent of saved implementation outputs.
    result = ForwardSolver(WrightFisherModel(Ne=100), UniformGrid(101),
                           SolverConfig(method="implicit_euler", dt=.5, output_times=[0, 1, 5])).solve(DeltaInitialCondition(.3))
    expected_H = .42*np.array([1, 1.0025**-2, 1.0025**-10])
    np.testing.assert_allclose(result.mean(), [.3, .3, .3], rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.heterozygosity(), expected_H, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.variance(), .21-expected_H/2, rtol=0, atol=1e-13)


def test_saved_independent_dense_distribution():
    reference = json.loads((Path(__file__).parent/"data"/"neutral_dense.json").read_text())
    result = ForwardSolver(WrightFisherModel(Ne=100), UniformGrid(101),
                           SolverConfig(output_times=reference["times"])).solve(DeltaInitialCondition(.3))
    error = np.abs(result.probability_mass-np.asarray(reference["probability_mass"])).sum(axis=1)
    assert error.max() < 1e-13
