"""Nodal mass projection; density arrays include both endpoints."""
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincc

from .config import finite_float, positive_float, positive_integer
from .coefficients import callable_metadata
from .exceptions import ConfigurationError, ResolutionWarning


@dataclass(frozen=True)
class UniformGrid:
    points: int = 2001
    x: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        points = positive_integer(self.points, "points", 101)
        x = np.linspace(0, 1, points, dtype=np.float64)
        x.flags.writeable = False
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "x", x)

    @property
    def dx(self):
        return 1 / (self.points - 1)


def validate_mass(values, points, normalize=False):
    if not isinstance(normalize, bool):
        raise ConfigurationError("normalize must be a boolean")
    try:
        p = np.array(values, dtype=np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("mass must be a numeric array") from exc
    if p.shape != (points,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ConfigurationError("mass must be finite, nonnegative, and match the grid")
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise ConfigurationError("initial mass must have a finite positive total")
    if normalize:
        p /= total
    elif abs(total-1) > 1e-12:
        raise ConfigurationError(f"initial mass sums to {total}; pass normalize=True explicitly")
    return p


@dataclass(frozen=True)
class DeltaInitialCondition:
    x0: float

    def __post_init__(self):
        x0 = finite_float(self.x0, "x0")
        if not 0 <= x0 <= 1:
            raise ConfigurationError("x0 must lie in [0,1]")
        object.__setattr__(self, "x0", x0)

    def to_mass(self, grid):
        p = np.zeros(grid.points)
        position = self.x0 / grid.dx
        i = min(int(position), grid.points-2)
        weight = np.clip(position-i, 0, 1)
        p[i], p[i+1] = 1-weight, weight
        return p

    def to_dict(self):
        return {"type": "delta", "x0": self.x0}


@dataclass(frozen=True)
class ArrayInitialCondition:
    values: object
    normalize: bool = False

    def to_mass(self, grid):
        return validate_mass(self.values, grid.points, self.normalize)

    def to_dict(self):
        return {"type": "array", "values": np.asarray(self.values).tolist(), "normalize": self.normalize}


def _warn_projection(p):
    endpoint = float(p[0]+p[-1])
    if endpoint > 1e-12:
        warnings.warn(f"continuous initial density projects {endpoint:.6g} mass onto endpoints; "
                      "refine the grid to reduce initial absorption error", ResolutionWarning, stacklevel=3)


@dataclass(frozen=True)
class DensityInitialCondition:
    values: object
    normalize: bool = False

    def to_mass(self, grid):
        p = np.zeros(grid.points)
        if callable(self.values):
            def f(x):
                value = finite_float(self.values(x), "density")
                if value < 0:
                    raise ConfigurationError("density must be nonnegative")
                return value
            for i, (left, right) in enumerate(zip(grid.x[:-1], grid.x[1:])):
                p[i] += quad(lambda x: (right-x)/grid.dx*f(x), left, right,
                             epsabs=1e-14, epsrel=1e-11)[0]
                p[i+1] += quad(lambda x: (x-left)/grid.dx*f(x), left, right,
                               epsabs=1e-14, epsrel=1e-11)[0]
        else:
            try:
                values = np.asarray(self.values, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ConfigurationError("density must be numeric") from exc
            if values.shape != (grid.points,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ConfigurationError("density array must be finite, nonnegative, and include endpoints")
            p[:-1] += grid.dx*(2*values[:-1]+values[1:])/6
            p[1:] += grid.dx*(values[:-1]+2*values[1:])/6
        p = validate_mass(p, grid.points, self.normalize)
        _warn_projection(p)
        return p

    def to_dict(self):
        values = callable_metadata(self.values) if callable(self.values) else np.asarray(self.values).tolist()
        return {"type": "density", "values": values, "normalize": self.normalize,
                "projection": "linear_hat"}


def _beta_interval(alpha, beta, left, right):
    c_left, c_right = betainc(alpha, beta, left), betainc(alpha, beta, right)
    return np.where(c_left > 0.5,
                    betaincc(alpha, beta, left)-betaincc(alpha, beta, right), c_right-c_left)


@dataclass(frozen=True)
class BetaInitialCondition:
    alpha: float
    beta: float

    def __post_init__(self):
        object.__setattr__(self, "alpha", positive_float(self.alpha, "alpha"))
        object.__setattr__(self, "beta", positive_float(self.beta, "beta"))

    def to_mass(self, grid):
        left, right = grid.x[:-1], grid.x[1:]
        mass = _beta_interval(self.alpha, self.beta, left, right)
        moment = self.alpha/(self.alpha+self.beta)*_beta_interval(self.alpha+1, self.beta, left, right)
        pleft = (right*mass-moment)/grid.dx
        pright = (moment-left*mass)/grid.dx
        # Use the symmetric variable near x=1 to reduce cancellation.
        tail = left > 0.5
        ymoment = self.beta/(self.alpha+self.beta)*_beta_interval(
            self.beta+1, self.alpha, 1-right[tail], 1-left[tail])
        pleft[tail] = (ymoment-(1-right[tail])*mass[tail])/grid.dx
        pright[tail] = ((1-left[tail])*mass[tail]-ymoment)/grid.dx
        if min(pleft.min(), pright.min()) < -1e-12:
            raise ConfigurationError("Beta projection lost precision; use less extreme shape parameters")
        p = np.zeros(grid.points)
        p[:-1] += np.maximum(pleft, 0)
        p[1:] += np.maximum(pright, 0)
        p = validate_mass(p, grid.points)
        _warn_projection(p)
        return p

    def to_dict(self):
        return {"type": "beta", "alpha": self.alpha, "beta": self.beta, "projection": "linear_hat"}
