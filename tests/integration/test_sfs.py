import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.linalg import expm
from scipy.special import comb

from wf_forward import ConfigurationError, calculate_sfs
from wf_forward.sfs import moment_generator


@pytest.mark.parametrize('n', [2, 3, 20, 100, 2140])
def test_neutral_equilibria_and_fold(n):
    result = calculate_sfs({'n': n, 'theta': 3})
    i = np.arange(1, n)
    np.testing.assert_allclose(result.derived, np.tile(3/i, (3,1)), rtol=2e-12)
    np.testing.assert_allclose(result.symmetric, np.tile(3*(1/i+1/(n-i)), (3,1)), rtol=2e-12)
    for mode in ('derived', 'symmetric'):
        spectrum = result.spectrum(mode)
        folded = result.spectrum(mode, folded=True)
        np.testing.assert_allclose(folded.sum(axis=1), spectrum.sum(axis=1))
        if n % 2 == 0:
            np.testing.assert_array_equal(folded[:, -1], spectrum[:, n//2-1])
        np.testing.assert_allclose(result.spectrum(mode, normalized=True).sum(axis=1), 1)
    assert result.metadata['max_relative_balance_error'] < 1e-12


def test_binomial_moment_operator_matches_continuous_diffusion():
    # Independent polynomial derivatives establish exact neutral moment closure.
    n = 9
    q = np.linspace(.01, .99, 23)
    polynomials = [comb(n,i)*Polynomial([0,1])**i*Polynomial([1,-1])**(n-i) for i in range(n+1)]
    B = np.array([p(q) for p in polynomials])
    expected = np.array([.5*q*(1-q)*p.deriv(2)(q) for p in polynomials])
    actual = moment_generator(n) @ B
    np.testing.assert_allclose(actual[1:-1], expected[1:-1], atol=1e-11)


def test_demographic_steps_match_independent_dense_exponential():
    n, ref = 12, 1000
    times = [0, 100, 200, 500, 2000]
    sizes = [1000, 100, 4000]
    result = calculate_sfs({'n':n, 'reference_Ne':ref,
        'history': {'times':[0,100,500], 'sizes':sizes, 'interpolation':'constant'},
        'output_times':times, 'rtol':1e-10, 'atol':1e-13})
    # Build independently, in generation units, including a constant affine source.
    y = np.r_[0, 1/np.arange(1,n), 0, 1.]
    expected = [y[1:n].copy()]
    for a,b in zip(times[:-1], times[1:]):
        Ne = sizes[0 if a < 100 else 1 if a < 500 else 2]
        A = np.zeros((n+2,n+2))
        for j in range(1,n):
            rate=j*(n-j)/(4*Ne)
            A[j-1,j]=A[j+1,j]=rate; A[j,j]=-2*rate
        A[1,-1]=n/(4*ref)
        y=expm(A*(b-a)) @ y
        expected.append(y[1:n].copy())
    np.testing.assert_allclose(result.derived, expected, rtol=2e-8, atol=1e-10)
    assert result.metadata['max_relative_balance_error'] < 1e-10


def test_linear_history_convergence_and_sample_projection():
    config={'n':20,'history':{'times':[0,500,1000], 'sizes':[10000,1000,100000]},
            'output_times':[0,250,500,750,1000]}
    coarse=calculate_sfs(config | {'rtol':1e-4,'atol':1e-7})
    fine=calculate_sfs(config | {'rtol':1e-8,'atol':1e-11})
    reference=calculate_sfs(config | {'rtol':1e-10,'atol':1e-13})
    assert np.max(abs(fine.derived-reference.derived)) < np.max(abs(coarse.derived-reference.derived))/5
    # Hypergeometric downprojection of the evolving 20-copy SFS to 2 copies
    # must equal the independently evolved two-copy system.
    small=calculate_sfs(config | {'n':2,'rtol':1e-10,'atol':1e-13})
    i=np.arange(1,20)
    pair_prob=2*i*(20-i)/(20*19)
    np.testing.assert_allclose(reference.derived @ pair_prob, small.derived[:,0], rtol=1e-8)


def test_initial_reference_scaling_single_output_and_theta_zero():
    result=calculate_sfs({'n':5,'reference_Ne':1000,'history':{'times':[0],'sizes':[2000]},'output_times':[5]})
    np.testing.assert_allclose(result.derived[0], 2/np.arange(1,5))
    zero=calculate_sfs({'theta':0,'output_times':[0]})
    assert np.all(zero.derived == 0)
    assert np.all(zero.spectrum(folded=True,normalized=True) == 0)


@pytest.mark.parametrize('change', [
    {'n':1},{'n':5001},{'n':True},{'theta':-1},{'theta':float('nan')},{'selection':.1},
    {'reference_Ne':0},{'output_times':[]},{'output_times':[1,0]},
    {'history':{'times':[1], 'sizes':[1000]}},
    {'history':{'times':[0,1], 'sizes':[1000]}},
    {'history':{'times':[0], 'sizes':[-1]}},
    {'history':{'times':[0], 'sizes':[1000], 'interpolation':'bad'}},
    {'history':{'times':[0,100000], 'sizes':[1000,10]}},
    {'rtol':1}, {'schema_version':True}
])
def test_invalid_sfs_config(change):
    with pytest.raises(ConfigurationError):
        calculate_sfs(change)


def test_sfs_cli_exports(tmp_path):
    config={'n':8,'theta':2,'output_times':[0,100]}
    (tmp_path/'config.json').write_text(json.dumps(config))
    command=[sys.executable,'-m','wf_forward','sfs',str(tmp_path/'config.json'),
             '--output',str(tmp_path/'sfs.npz'),'--csv',str(tmp_path/'sfs.csv')]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode == 0,run.stderr
    with np.load(tmp_path/'sfs.npz',allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays['derived'],calculate_sfs(config).derived)
        assert json.loads(str(arrays['metadata_json']))['config']['n']==8
    rows=np.genfromtxt(tmp_path/'sfs.csv',delimiter=',',names=True)
    assert len(rows)==14
    np.testing.assert_allclose(rows['derived_expected_sites'][:7],2/np.arange(1,8))
