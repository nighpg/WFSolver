"""Sparse exponential and probability-preserving implicit Euler integration."""
from collections import OrderedDict
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.sparse.linalg import expm_multiply

from .exceptions import NumericalError, ResolutionWarning
from .operator import build_generator


class OperatorProvider:
    def __init__(self, model, grid):
        self.model, self.grid = model, grid
        self.cache = OrderedDict()
        self.max_peclet = 0.0
        self.build_seconds = 0.0
        self.build_count = 0
        self.warned = False
        self.timings = {"coefficient_evaluation_seconds": 0.0, "matrix_assembly_seconds": 0.0}

    def get(self, t):
        started = perf_counter()
        key = self.model.operator_key(t)
        self.timings["coefficient_evaluation_seconds"] += perf_counter()-started
        if key not in self.cache:
            start = perf_counter()
            operator = build_generator(self.model, self.grid, t, self.timings)
            self.build_seconds += perf_counter()-start
            self.build_count += 1
            self.max_peclet = max(self.max_peclet, operator.max_peclet)
            if self.max_peclet > 2 and not self.warned:
                warnings.warn(f"max interior Peclet={self.max_peclet:.6g} > 2; refine the grid",
                              ResolutionWarning, stacklevel=3)
                self.warned = True
            self.cache[key] = operator
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return key, self.cache[key]


class ImplicitStepper:
    def __init__(self, provider):
        self.provider = provider
        self.factors = OrderedDict()
        self.factorizations = 0
        self.linear_solves = 0
        self.factor_seconds = 0.0
        self.solve_seconds = 0.0

    def advance(self, p, t, dt, out=None):
        operator_key, q = self.provider.get(t + dt/2)
        key = (operator_key, dt)
        if key not in self.factors:
            start = perf_counter()
            # Q.T lower[i] = up[i], upper[i] = down[i+1].
            lower = -dt*q.up[:-1]
            diagonal = 1-dt*q.diagonal
            upper = -dt*q.down[1:]
            if not np.all(np.isfinite(diagonal)):
                raise NumericalError("implicit matrix overflow; reduce dt")
            factor = dgttrf(lower, diagonal, upper, overwrite_dl=True,
                           overwrite_d=True, overwrite_du=True)
            if factor[-1] != 0:
                raise NumericalError(f"tridiagonal factorization failed: LAPACK info={factor[-1]}")
            self.factors[key] = factor[:-1]
            self.factorizations += 1
            self.factor_seconds += perf_counter()-start
            if len(self.factors) > 4:
                self.factors.popitem(last=False)
        self.factors.move_to_end(key)
        start = perf_counter()
        # LAPACK's tridiagonal solve supports multiple RHS; expose one column.
        if out is None:
            out = np.empty_like(p)
        np.copyto(out, p)
        rhs = out.reshape(-1, 1)
        result, info = dgttrs(*self.factors[key], rhs, overwrite_b=True)
        self.solve_seconds += perf_counter()-start
        self.linear_solves += 1
        if info != 0:
            raise NumericalError(f"tridiagonal solve failed: LAPACK info={info}")
        return result[:, 0]


def integrate_expm(p, times, provider, diagnostics, config):
    _, q = provider.get(0)
    A = q.as_sparse().T.tocsr()
    trace = float(q.diagonal.sum())
    calls = 0
    if len(times) > 1 and np.array_equal(times, np.linspace(times[0], times[-1], len(times))):
        P = expm_multiply(A, p, start=times[0], stop=times[-1], num=len(times), traceA=trace)
        calls = 1
        for row in P:
            diagnostics.check(row, config)
    else:
        P = np.empty((len(times), len(p)))
        previous = 0.0
        for i, t in enumerate(times):
            dt = t-previous
            if dt > 0:
                p = expm_multiply(A*dt, p, traceA=trace*dt)
                calls += 1
            diagnostics.check(p, config)
            P[i] = p
            previous = t
    return P, {"steps": None, "expm_calls": calls, "internal_steps_available": False,
               "min_dt": None, "max_dt": None, "rejected_steps": 0,
               "factorizations": 0, "linear_solves": 0}


def integrate_implicit(p, times, provider, diagnostics, config):
    stepper = ImplicitStepper(provider)
    P = np.empty((len(times), len(p)))
    events = np.unique(np.concatenate((times, np.asarray(provider.model.all_breakpoints()))))
    events = events[(events > 0) & (events <= times[-1])]
    positive_gaps = np.diff(np.concatenate(([0.0], times)))
    positive_gaps = positive_gaps[positive_gaps > 0]
    h = config.dt or (float(positive_gaps.min())/10 if len(positive_gaps) else 1.0)
    if config.dt is None and config.parameter_timescale is not None:
        h = min(h, config.parameter_timescale/20)
    if config.max_dt is not None:
        h = min(h, config.max_dt)
    h = max(h, config.min_dt)
    t, out_index, attempts, accepted, rejections = 0.0, 0, 0, 0, 0
    min_dt, max_dt = np.inf, 0.0
    coarse_buffer, half_buffer, fine_buffer, error_buffer = (np.empty_like(p) for _ in range(4))
    if times[0] == 0:
        P[0] = p
        out_index = 1
    for target in events:
        while t < target:
            remaining = target-t
            rounding = 8*np.finfo(float).eps*max(abs(target), abs(t), h)
            finishes = remaining <= h+rounding
            dt = remaining if finishes else h
            if finishes and abs(remaining-h) <= rounding:
                dt = h
            if t+dt == t or dt <= 0:
                raise NumericalError("time step cannot advance time at float64 precision")
            attempts += 1
            if attempts > config.max_steps:
                raise NumericalError("max_steps exceeded")
            coarse = diagnostics.check(stepper.advance(p, t, dt, coarse_buffer), config)
            if config.dt is None:
                half = diagnostics.check(stepper.advance(p, t, dt/2, half_buffer), config)
                fine = diagnostics.check(stepper.advance(half, t+dt/2, dt/2, fine_buffer), config)
                np.subtract(fine, coarse, out=error_buffer)
                np.abs(error_buffer, out=error_buffer)
                error = float(error_buffer.sum())
                tolerance = config.atol + config.rtol*float(np.abs(fine).sum())
                if error > tolerance:
                    rejections += 1
                    if rejections > config.max_rejections or dt/2 < config.min_dt:
                        raise NumericalError("adaptive tolerance cannot be met within step limits")
                    h = dt/2
                    continue
                p, fine_buffer = fine, p
                accepted += 2
                min_dt, max_dt = min(min_dt, dt/2), max(max_dt, dt/2)
                # Do not let a tiny event-alignment step erase the learned step size.
                proposed = dt*2 if error < tolerance/4 else dt
                h = max(h, proposed) if dt < h else proposed
                if config.max_dt is not None:
                    h = min(h, config.max_dt)
            else:
                p, coarse_buffer = coarse, p
                accepted += 1
                min_dt, max_dt = min(min_dt, dt), max(max_dt, dt)
            t = float(target) if finishes else t+dt
        if out_index < len(times) and target == times[out_index]:
            P[out_index] = p
            out_index += 1
    return P, {"steps": accepted, "attempted_steps": attempts,
               "internal_steps_available": True, "rejected_steps": rejections,
               "min_dt": float(min_dt) if accepted else None,
               "max_dt": float(max_dt) if accepted else None,
               "factorizations": stepper.factorizations, "linear_solves": stepper.linear_solves,
               "factorization_seconds": stepper.factor_seconds, "linear_solve_seconds": stepper.solve_seconds}
