"""Neutral infinite-sites sample spectra with explicit mutation input.

The binomial moments of the neutral diffusion close exactly at sample size n.
This is a sample-count moment system, not a discretized population density.
Time is in generations; theta is the reference equilibrium amplitude.
"""
import csv
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .config import finite_float, positive_float, positive_integer
from .exceptions import ConfigurationError, NumericalError
from .io import atomic_file, dumps


def validate_sfs_config(data):
    if not isinstance(data, dict):
        raise ConfigurationError('SFS configuration must be an object')
    allowed = {'schema_version', 'n', 'theta', 'reference_Ne', 'history', 'output_times', 'rtol', 'atol'}
    if set(data)-allowed:
        raise ConfigurationError('Unknown SFS settings: '+', '.join(sorted(set(data)-allowed)))
    if type(data.get('schema_version', 1)) is not int or data.get('schema_version', 1) != 1:
        raise ConfigurationError('Unknown SFS schema version')
    n = positive_integer(data.get('n', 100), 'n', 2)
    if n > 5000:
        raise ConfigurationError('SFS supports at most 5000 sampled allele copies')
    theta = finite_float(data.get('theta', 1), 'theta')
    if theta < 0 or theta > 1e12:
        raise ConfigurationError('theta must be between 0 and 1e12')
    reference = positive_float(data.get('reference_Ne', 10000), 'reference_Ne')
    history = data.get('history', {'times': [0], 'sizes': [reference], 'interpolation': 'linear'})
    if not isinstance(history, dict) or set(history)-{'times', 'sizes', 'interpolation'}:
        raise ConfigurationError('history must contain times, sizes, and interpolation')
    try:
        times = [finite_float(t, 'history time') for t in history.get('times', [])]
        sizes = [positive_float(v, 'population size') for v in history.get('sizes', [])]
        output = [finite_float(t, 'output time') for t in data.get('output_times', [0, 1000, 10000])]
    except TypeError as exc:
        raise ConfigurationError('SFS times and sizes must be arrays') from exc
    if (not times or len(times) > 201 or times[0] != 0 or len(times) != len(sizes)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ConfigurationError('Use 1–201 increasing history times starting at 0, with one size per time')
    if (not output or len(output) > 201 or output[0] < 0
            or any(b <= a for a, b in zip(output, output[1:])) or times[-1] > output[-1]):
        raise ConfigurationError('Use 1–201 increasing nonnegative output times covering every history knot')
    interpolation = history.get('interpolation', 'linear')
    if interpolation not in ('linear', 'constant'):
        raise ConfigurationError('history interpolation must be linear or constant')
    rtol = positive_float(data.get('rtol', 1e-7), 'rtol')
    atol = positive_float(data.get('atol', 1e-10), 'atol')
    if not 1e-12 <= rtol <= 1e-3 or not 1e-14 <= atol <= 1e-5:
        raise ConfigurationError('Use rtol in [1e-12,1e-3] and atol in [1e-14,1e-5]')
    # Keep scaled time, rates, and output intensities in a well-conditioned range.
    if not 1 <= reference <= 1e12 or any(not 1 <= v <= 1e12 for v in sizes) or output[-1] > 1e12:
        raise ConfigurationError('Population sizes must be in [1,1e12] and time at most 1e12 generations')
    return {'schema_version': 1, 'n': n, 'theta': theta, 'reference_Ne': reference,
            'history': {'times': times, 'sizes': sizes, 'interpolation': interpolation},
            'output_times': output, 'rtol': rtol, 'atol': atol}


def moment_generator(n):
    """Generator transpose in units of 2*Nref generations, before /rho.

    Interior row i: r_(i-1)y_(i-1)-2*r_i*y_i+r_(i+1)y_(i+1),
    r_i=i(n-i)/2. Endpoints accumulate removal from the sample's polymorphic
    classes; they are not population loss/fixation probabilities.
    """
    i = np.arange(n+1, dtype=float)
    rates = i*(n-i)/2
    return diags((rates[:-1], -2*rates, rates[1:]), (-1, 0, 1), format='csc')


@dataclass(frozen=True)
class SFSResult:
    times: np.ndarray
    derived: np.ndarray  # Expected sites, shape (T,n-1), includes theta.
    metadata: dict

    @property
    def symmetric(self):
        return self.derived+self.derived[:, ::-1]

    def spectrum(self, mode='symmetric', folded=False, normalized=False):
        if mode not in ('derived', 'symmetric'):
            raise ConfigurationError('SFS mode must be derived or symmetric')
        values = (self.derived if mode == 'derived' else self.symmetric).copy()
        n = values.shape[1]+1
        if folded:
            folded_values = values[:, :n//2].copy()
            pairs = (n-1)//2
            folded_values[:, :pairs] += values[:, -pairs:][:, ::-1] if pairs else 0
            values = folded_values
        if normalized:
            total = values.sum(axis=1, keepdims=True)
            values = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
        return values

    def to_dict(self):
        return {'times': self.times, 'derived': self.derived, 'symmetric': self.symmetric,
                'metadata': self.metadata}

    def save_npz(self, path):
        with atomic_file(path, binary=True) as stream:
            np.savez_compressed(stream, times=self.times, derived=self.derived,
                                symmetric=self.symmetric, metadata_json=np.asarray(dumps(self.metadata)))

    def to_csv(self, path):
        with atomic_file(path) as stream:
            writer = csv.writer(stream)
            writer.writerow(('time', 'derived_allele_count', 'n', 'derived_expected_sites', 'symmetric_expected_sites'))
            for t, row, symmetric in zip(self.times, self.derived, self.symmetric):
                writer.writerows((t, i, len(row)+1, a, b) for i, (a, b) in enumerate(zip(row, symmetric), 1))


def calculate_sfs(config=None):
    """Evolve a neutral ancestral equilibrium through a declarative size history.

    theta=1 gives g_i=1/i at Ne=Nref. A symmetric spectrum is the sum of
    that spectrum and its reflection (not their average). Both outputs exclude
    monomorphic classes. Selection and finite-site mutation are not used.
    """
    started = perf_counter()
    config = validate_sfs_config({} if config is None else config)
    n, ref, theta = config['n'], config['reference_Ne'], config['theta']
    history = config['history']
    knots = np.asarray(history['times'])/(2*ref)
    sizes = np.asarray(history['sizes'])/ref
    times = np.asarray(config['output_times'])
    outputs = times/(2*ref)
    generator = moment_generator(n)
    source = np.zeros(n+1); source[1] = n/2
    initial = np.zeros(n+1); initial[1:-1] = sizes[0]/np.arange(1, n)
    state = initial.copy()
    saved = np.empty((len(times), n+1))
    if outputs[0] == 0:
        saved[0] = state
    edges = np.unique(np.r_[0, knots, outputs[-1]])
    nfev = nlu = 0
    minimum = 0.0
    correction = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        segment = np.searchsorted(knots, left, side='right')-1
        def rho(t):
            if history['interpolation'] == 'constant':
                return sizes[segment]
            return np.interp(t, knots, sizes)
        def rhs(t, y):
            return (generator @ y)/rho(t)+source
        def jac(t, y):
            return generator/rho(t)
        indices = np.flatnonzero((outputs > left) & (outputs <= right))
        evaluation = np.unique(np.r_[outputs[indices], right])
        solution = solve_ivp(rhs, (left, right), state, method='BDF', jac=jac,
                             t_eval=evaluation, rtol=config['rtol'], atol=config['atol'])
        if not solution.success or not np.isfinite(solution.y).all():
            raise NumericalError('SFS integration failed: '+solution.message)
        minimum = min(minimum, float(solution.y.min()))
        if minimum < -10*config['atol']:
            raise NumericalError('SFS integration produced negative expected site counts; tighten tolerances')
        correction += float(-np.minimum(solution.y, 0).sum())
        values = np.maximum(solution.y, 0)
        saved[indices] = values[:, np.searchsorted(evaluation, outputs[indices])].T
        state = values[:, -1]
        nfev += solution.nfev; nlu += solution.nlu
    expected_total = initial.sum()+outputs*n/2
    balance = float(np.max(np.abs(saved.sum(axis=1)-expected_total)/np.maximum(1, expected_total)))
    if balance > max(1e-8, 10*config['rtol']):
        raise NumericalError('SFS mutation-input / sample-boundary-outflow balance failed')
    derived = saved[:, 1:-1]*theta
    derived.flags.writeable = times.flags.writeable = False
    from .solver import environment_metadata
    metadata = {'schema_version': 1, 'environment': environment_metadata(), 'config': config, 'method': 'neutral_binomial_moments_bdf',
                'elapsed_seconds': perf_counter()-started, 'time_unit': 'generation',
                'initial_condition': 'ancestral_neutral_equilibrium',
                'theta_convention': 'reference equilibrium amplitude: derived theta/i; symmetric theta*(1/i+1/(n-i))',
                'symmetric_convention': 'sum of derived spectrum and reflection; two-sided input, not an average',
                'source_per_unit_theta_per_generation': n/(4*ref),
                'minimum_before_correction_per_theta': minimum,
                'clipped_mass_per_theta': correction, 'max_relative_balance_error': balance,
                'sample_boundary_outflow_per_theta': saved[:, [0, -1]].tolist(),
                'rhs_evaluations': nfev, 'factorizations': nlu,
                'limitations': 'Neutral infinite-sites model. Symmetric demographic extension uses equal mirrored mutation sources; the PDF excerpt does not uniquely specify its boundary sources.'}
    return SFSResult(times, derived, metadata)
