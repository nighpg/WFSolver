"""Diploid coefficients in generations: variance V=x(1-x)/(2 Ne)."""
from dataclasses import dataclass, fields
from typing import Callable
import hashlib
import inspect
import warnings

import numpy as np

from .config import finite_float
from .exceptions import ConfigurationError, ModelConsistencyWarning
from .schedules import PiecewiseConstant

ScalarOrTimeFunction = float | Callable[[float], float]
PARAMETERS = ("Ne", "selection", "dominance", "mutation_forward", "mutation_backward")


def callable_metadata(value, label=None):
    result = {"type": "callable", "qualified_name":
              f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', type(value).__name__)}",
              "label": label, "reconstructible": False}
    try:
        result["source_sha256"] = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
    except (OSError, TypeError):
        result["source_sha256"] = None
    return result


@dataclass(frozen=True)
class WrightFisherModel:
    """Weak selection with w_aa=1, w_Aa=1+h*s, w_AA=1+s.

    All rates and time arguments use generations. drift_override replaces the
    entire drift, including mutation, and accepts (x, t).
    """
    Ne: ScalarOrTimeFunction = 10_000.0
    selection: ScalarOrTimeFunction = 0.0
    dominance: ScalarOrTimeFunction = 0.5
    mutation_forward: ScalarOrTimeFunction = 0.0
    mutation_backward: ScalarOrTimeFunction = 0.0
    boundary: str = "absorbing"
    drift_override: Callable | None = None
    time_homogeneous: bool | None = None
    breakpoints: tuple[float, ...] = ()
    label: str | None = None
    time_unit: str = "generation"
    ploidy: int = 2

    def __post_init__(self):
        if self.boundary not in ("absorbing", "mutation"):
            raise ConfigurationError("boundary must be absorbing or mutation")
        if self.time_unit != "generation" or self.ploidy != 2 or isinstance(self.ploidy, bool):
            raise ConfigurationError("v1 supports explicit generation units and diploid ploidy=2 only")
        if self.time_homogeneous is not None and not isinstance(self.time_homogeneous, bool):
            raise ConfigurationError("time_homogeneous must be a bool or None")
        if self.label is not None and not isinstance(self.label, str):
            raise ConfigurationError("label must be a string or None")
        if self.drift_override is not None and not callable(self.drift_override):
            raise ConfigurationError("drift_override must be callable")
        for name in PARAMETERS:
            value = getattr(self, name)
            if isinstance(value, PiecewiseConstant):
                samples = value.values
                if self.time_homogeneous is True and not value.is_constant:
                    raise ConfigurationError("a varying schedule cannot be declared time homogeneous")
            elif callable(value):
                samples = ()
            else:
                value = finite_float(value, name)
                object.__setattr__(self, name, value)
                samples = (value,)
            for sample in samples:
                self._validate_parameter(name, sample)
        try:
            bp = tuple(finite_float(t, "breakpoint") for t in self.breakpoints)
        except TypeError as exc:
            raise ConfigurationError("breakpoints must be a sequence") from exc
        if any(t < 0 for t in bp) or any(b <= a for a, b in zip(bp, bp[1:])):
            raise ConfigurationError("breakpoints must be nonnegative and strictly increasing")
        object.__setattr__(self, "breakpoints", bp)
        self.parameter_values(0)
        if self.boundary == "mutation" and all(
            not callable(getattr(self, n)) and getattr(self, n) == 0
            for n in ("mutation_forward", "mutation_backward")
        ) and self.drift_override is None:
            warnings.warn("mutation boundary with zero mutation is absorbing", ModelConsistencyWarning)

    def _validate_parameter(self, name, value):
        value = finite_float(value, name)
        if name == "Ne" and value <= 0:
            raise ConfigurationError("Ne must be positive")
        if name.startswith("mutation_"):
            if value < 0:
                raise ConfigurationError("mutation rates must be nonnegative")
            if self.boundary == "absorbing" and value > 0:
                raise ConfigurationError("absorbing boundary requires zero mutation rates")

    def parameter_values(self, t):
        values = []
        for name in PARAMETERS:
            parameter = getattr(self, name)
            try:
                value = parameter(t) if callable(parameter) else parameter
            except Exception as exc:
                raise ConfigurationError(f"failed evaluating {name} at t={t}") from exc
            self._validate_parameter(name, value)
            values.append(float(value))
        return tuple(values)

    def evaluate(self, x, t):
        Ne, s, h, u, v = self.parameter_values(t)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            variance = (x * (1 - x) / Ne) / 2
            if self.drift_override is None:
                drift = s*x*(1-x)*(h+(1-2*h)*x) + u*(1-x)-v*x
            else:
                try:
                    drift = np.asarray(self.drift_override(x.copy(), t), dtype=np.float64)
                except Exception as exc:
                    raise ConfigurationError(f"failed evaluating drift_override at t={t}") from exc
        if np.shape(drift) != np.shape(x) or not np.all(np.isfinite(drift)):
            raise ConfigurationError("drift must return a finite array with the same shape as x")
        if not np.all(np.isfinite(variance)) or np.any(variance < 0):
            raise ConfigurationError("variance must be finite and nonnegative")
        return drift, variance

    def drift(self, x, t=0.0):
        return self.evaluate(np.asarray(x, dtype=np.float64), t)[0]

    def variance(self, x, t=0.0):
        return self.evaluate(np.asarray(x, dtype=np.float64), t)[1]

    def is_time_homogeneous(self):
        if self.time_homogeneous is not None:
            return self.time_homogeneous
        return self.drift_override is None and all(
            v.is_constant if isinstance(v, PiecewiseConstant) else not callable(v)
            for v in (getattr(self, n) for n in PARAMETERS)
        )

    def all_breakpoints(self):
        bp = set(self.breakpoints)
        for name in PARAMETERS:
            value = getattr(self, name)
            if isinstance(value, PiecewiseConstant):
                bp.update(value.breakpoints)
        return tuple(sorted(bp))

    def operator_key(self, t):
        if self.is_time_homogeneous():
            return ("constant",)
        if self.drift_override is None:
            return self.parameter_values(t)
        return ("time", t)

    def to_dict(self):
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, PiecewiseConstant):
                value = value.to_dict()
            elif callable(value):
                value = callable_metadata(value, self.label)
            elif isinstance(value, tuple):
                value = list(value)
            result[field.name] = value
        return result
