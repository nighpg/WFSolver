"""Validated numerical configuration (times are measured from t=0)."""
from dataclasses import dataclass
import math
from numbers import Real, Integral

from .exceptions import ConfigurationError


def finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite real number")
    return float(value)


def positive_float(value, name):
    value = finite_float(value, name)
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class SolverConfig:
    output_times: tuple[float, ...] = (0.0, 100.0, 1000.0)
    method: str = "auto"
    dt: float | None = None
    atol: float = 1e-8
    rtol: float = 1e-5
    min_dt: float = 1e-12
    max_dt: float | None = None
    parameter_timescale: float | None = None
    max_steps: int = 1_000_000
    max_rejections: int = 1000
    probability_tolerance: float = 1e-10
    negative_clip_tolerance: float = 1e-14
    negative_error_tolerance: float = 1e-10
    negative_policy: str = "error"
    max_output_bytes: int = 512 * 1024**2
    spatial_scheme: str = "upwind_ctmc"
    dtype: str = "float64"

    def __post_init__(self):
        try:
            times = tuple(finite_float(t, "output time") for t in self.output_times)
        except TypeError as exc:
            raise ConfigurationError("output_times must be a sequence") from exc
        if not times or times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ConfigurationError("output_times must be nonnegative and strictly increasing")
        object.__setattr__(self, "output_times", times)
        if self.method not in ("auto", "expm", "implicit_euler"):
            raise ConfigurationError("method must be auto, expm, or implicit_euler")
        if self.spatial_scheme != "upwind_ctmc" or self.dtype != "float64":
            raise ConfigurationError("v1 supports upwind_ctmc and float64 only")
        for name in ("atol", "rtol", "min_dt", "probability_tolerance",
                     "negative_clip_tolerance", "negative_error_tolerance"):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))
        for name in ("dt", "max_dt", "parameter_timescale"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, positive_float(value, name))
        for name in ("max_steps", "max_rejections", "max_output_bytes"):
            object.__setattr__(self, name, positive_integer(getattr(self, name), name))
        if self.max_dt is not None and self.max_dt < self.min_dt:
            raise ConfigurationError("max_dt must be >= min_dt")
        if self.dt is not None and self.dt < self.min_dt:
            raise ConfigurationError("dt must be >= min_dt")
        if self.negative_clip_tolerance > self.negative_error_tolerance:
            raise ConfigurationError("negative_clip_tolerance must not exceed negative_error_tolerance")
        if self.negative_policy not in ("error", "warn"):
            raise ConfigurationError("negative_policy must be error or warn")
