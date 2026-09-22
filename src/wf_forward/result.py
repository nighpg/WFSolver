"""Simulation output. Endpoint aliases denote occupancy under mutation."""
from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np

from .config import SolverConfig
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class SimulationResult:
    times: np.ndarray
    x: np.ndarray
    probability_mass: np.ndarray
    metadata: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self):
        for name in ("times", "x", "probability_mass"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ConfigurationError(f"result {name} contains non-finite values")
            object.__setattr__(self, name, value)
        SolverConfig(output_times=self.times)
        if self.x.ndim != 1 or len(self.x) < 101 or self.x[0] != 0 or self.x[-1] != 1:
            raise ConfigurationError("result grid must include 0 and 1 and have at least 101 points")
        if not np.allclose(np.diff(self.x), 1/(len(self.x)-1), rtol=1e-10, atol=1e-15):
            raise ConfigurationError("v1 result grid must be uniform")
        if self.probability_mass.shape != (len(self.times), len(self.x)) or np.any(self.probability_mass < 0):
            raise ConfigurationError("result mass has invalid shape or negative entries")
        if not isinstance(self.metadata, Mapping) or not isinstance(self.diagnostics, Mapping):
            raise ConfigurationError("metadata and diagnostics must be mappings")
        for value in (self.times, self.x, self.probability_mass):
            value.flags.writeable = False

    @property
    def density(self):
        return self.probability_mass[:, 1:-1]*(len(self.x)-1)

    @property
    def p_at_zero(self):
        return self.probability_mass[:, 0]

    @property
    def p_at_one(self):
        return self.probability_mass[:, -1]

    @property
    def p_loss(self):
        """Endpoint occupancy; cumulative loss only for absorbing boundaries."""
        return self.p_at_zero

    @property
    def p_fix(self):
        """Endpoint occupancy; cumulative fixation only for absorbing boundaries."""
        return self.p_at_one

    @property
    def p_segregating(self):
        return self.probability_mass[:, 1:-1].sum(axis=1)

    def mean(self):
        return self.probability_mass @ self.x

    def variance(self):
        # Endpoint mass is included; rounding alone may make the variance tiny negative.
        return np.maximum(0, self.probability_mass @ (self.x*self.x) - self.mean()**2)

    def heterozygosity(self):
        return self.probability_mass @ (2*self.x*(1-self.x))

    def save_npz(self, path):
        from .io import save_npz
        save_npz(self, path)

    @classmethod
    def load_npz(cls, path):
        from .io import load_npz
        return load_npz(path)

    def to_summary_csv(self, path):
        from .io import save_summary_csv
        save_summary_csv(self, path)

    def to_distribution_csv(self, path, time_index=-1):
        from .io import save_distribution_csv
        save_distribution_csv(self, path, time_index)

    def sample_trajectories(self, paths=32, seed=42, max_events=5_000_000):
        from .trajectories import sample_trajectories
        return sample_trajectories(self, paths, seed, max_events)
