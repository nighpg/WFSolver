import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from wf_forward import (ConfigurationError, DeltaInitialCondition, ForwardSolver,
                        NumericalError, PiecewiseConstant, SimulationResult,
                        SolverConfig, UniformGrid, WrightFisherModel)
from wf_forward.operator import build_generator


def solve(model=None, times=(0, 1, 2), x0=.3):
    return ForwardSolver(model or WrightFisherModel(Ne=100), UniformGrid(101),
                         SolverConfig(output_times=times)).solve(DeltaInitialCondition(x0))


def test_sampling_seed_and_exports(tmp_path):
    result = solve()
    result.save_npz(tmp_path/'source.npz')
    sample = result.sample_trajectories(paths=100, seed=19)
    np.testing.assert_array_equal(sample.frequencies, SimulationResult.load_npz(
        tmp_path/'source.npz').sample_trajectories(paths=100, seed=19).frequencies)
    assert not np.array_equal(sample.frequencies, result.sample_trajectories(paths=100, seed=20).frequencies)
    sample.save_npz(tmp_path/'paths.npz')
    sample.to_csv(tmp_path/'paths.csv')
    with np.load(tmp_path/'paths.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['frequencies'], sample.frequencies)
        assert json.loads(str(saved['metadata_json']))['seed'] == 19
    csv = np.genfromtxt(tmp_path/'paths.csv', delimiter=',', names=True)
    np.testing.assert_array_equal(csv['allele_frequency'].reshape(100, 3), sample.frequencies)
    run = subprocess.run([sys.executable, '-m', 'wf_forward', 'sample', str(tmp_path/'source.npz'),
                          '--paths', '100', '--seed', '19', '--output', str(tmp_path/'cli.npz')],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    with np.load(tmp_path/'cli.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['frequencies'], sample.frequencies)


@pytest.mark.parametrize('endpoint', [0, 1])
def test_absorbing_paths_remain_absorbed(endpoint):
    samples = solve(x0=endpoint).sample_trajectories(paths=10)
    assert np.all(samples.frequencies == endpoint)
    assert samples.metadata['events'] == 0


def test_temporal_dependence_and_marginals():
    # Compare the joint second moment too: independent marginal draws would fail.
    result = solve(times=(0, 5, 10))
    sample = result.sample_trajectories(paths=10000, seed=2026)
    x = sample.frequencies
    for i in (1, 2):
        p = result.probability_mass[i]
        for observable in (result.x, result.x**2, (result.x < .3).astype(float)):
            expected = p @ observable
            observed = observable[np.rint(x[:, i]*100).astype(int)].mean()
            se = np.sqrt(max(0, p @ observable**2-expected**2)/len(x))
            assert abs(observed-expected) < 5*se+1e-10
    # Neutral martingale: E[X(t1)*X(t2)] = E[X(t1)^2].
    joint = x[:, 1]*x[:, 2]
    assert abs(joint.mean()-result.probability_mass[1] @ result.x**2) < 5*joint.std()/np.sqrt(len(x))
    assert np.corrcoef(x[:, 1], x[:, 2])[0, 1] > .6


def test_mutation_schedule_reactivates_endpoint_and_matches_exact_reference():
    model = WrightFisherModel(Ne=100, mutation_forward=PiecewiseConstant([0, 1], [0, .005]),
                             mutation_backward=.002, boundary='mutation')
    result = solve(model, times=(0, .5, 1, 2, 4), x0=0)
    sample = result.sample_trajectories(paths=10000, seed=12)
    assert np.all(sample.frequencies[:, :3] == 0)
    q = build_generator(model, UniformGrid(101), 1).as_sparse().T
    for index, duration in ((3, 1), (4, 3)):
        expected = expm_multiply(q*duration, result.probability_mass[0])
        y = sample.frequencies[:, index]
        mean, variance = expected @ result.x, expected @ result.x**2-(expected @ result.x)**2
        assert abs(y.mean()-mean) < 5*np.sqrt(variance/len(y))
        p0 = expected[0]
        assert abs(np.mean(y == 0)-p0) < 5*np.sqrt(p0*(1-p0)/len(y))
    assert np.any(sample.frequencies[:, -1] > 0)


def test_sampling_starts_at_first_saved_marginal():
    result = solve(times=(5,))
    sample = result.sample_trajectories(paths=10000)
    assert sample.times.tolist() == [5]
    assert sample.metadata['events'] == 0
    assert sample.frequencies.std() > .05
    assert abs(sample.frequencies.mean()-.3) < .003


@pytest.mark.parametrize('options', [{'paths': 0}, {'paths': True}, {'seed': -1}, {'seed': 1.5}, {'max_events': 0}])
def test_sampling_invalid_options(options):
    with pytest.raises(ConfigurationError):
        solve().sample_trajectories(**options)


def test_sampling_event_limit_and_callable():
    with pytest.raises(NumericalError, match='event limit'):
        solve().sample_trajectories(max_events=1)
    with pytest.raises(ConfigurationError, match='callables'):
        solve(WrightFisherModel(Ne=lambda t: 100)).sample_trajectories()
