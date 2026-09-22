"""Declarative, right-continuous parameter schedules."""
from bisect import bisect_right
from dataclasses import dataclass

from .config import finite_float
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class PiecewiseConstant:
    breakpoints: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self):
        try:
            bp = tuple(finite_float(t, "breakpoint") for t in self.breakpoints)
            values = tuple(finite_float(v, "schedule value") for v in self.values)
        except TypeError as exc:
            raise ConfigurationError("schedule breakpoints and values must be sequences") from exc
        if not bp or bp[0] != 0 or any(b <= a for a, b in zip(bp, bp[1:])):
            raise ConfigurationError("breakpoints must start at 0 and strictly increase")
        if len(bp) != len(values):
            raise ConfigurationError("one schedule value is required per breakpoint")
        object.__setattr__(self, "breakpoints", bp)
        object.__setattr__(self, "values", values)

    def __call__(self, t):
        t = finite_float(t, "time")
        if t < 0:
            raise ConfigurationError("time must be nonnegative")
        return self.values[bisect_right(self.breakpoints, t) - 1]

    @property
    def is_constant(self):
        return len(set(self.values)) == 1

    def to_dict(self):
        return {"type": "piecewise_constant", "breakpoints": list(self.breakpoints),
                "values": list(self.values)}
