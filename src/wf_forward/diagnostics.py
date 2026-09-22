"""Inspect raw mass before performing any permitted roundoff correction."""
from dataclasses import dataclass, asdict
from time import perf_counter
import warnings

import numpy as np

from .exceptions import NumericalError


@dataclass
class ProbabilityDiagnostics:
    max_probability_error: float = 0.0
    minimum_probability: float = 1.0
    correction_count: int = 0
    corrected_entries: int = 0
    max_negative_corrected: float = 0.0
    total_negative_mass_corrected: float = 0.0
    total_correction_l1: float = 0.0
    checks: int = 0
    diagnostic_seconds: float = 0.0

    def check(self, p, config):
        started = perf_counter()
        try:
            return self._check(p, config)
        finally:
            self.diagnostic_seconds += perf_counter()-started

    def _check(self, p, config):
        if not np.all(np.isfinite(p)):
            raise NumericalError("non-finite probability")
        total, minimum = float(p.sum()), float(p.min())
        error = abs(total-1)
        self.max_probability_error = max(self.max_probability_error, error)
        self.minimum_probability = min(self.minimum_probability, minimum)
        self.checks += 1
        if error > config.probability_tolerance:
            raise NumericalError(f"probability error before correction {error:.6g} exceeds tolerance")
        if minimum < -config.negative_error_tolerance:
            raise NumericalError(f"negative probability {minimum:.6g}")
        if minimum < -config.negative_clip_tolerance:
            if config.negative_policy == "error":
                raise NumericalError(f"negative probability {minimum:.6g} exceeds clipping tolerance")
            warnings.warn(f"correcting negative probability {minimum:.6g}", RuntimeWarning, stacklevel=2)
        if minimum < 0:
            before = p.copy()
            negative = p < 0
            self.corrected_entries += int(negative.sum())
            self.total_negative_mass_corrected += float(-p[negative].sum())
            self.max_negative_corrected = max(self.max_negative_corrected, -minimum)
            p[negative] = 0
            p /= p.sum()
            self.total_correction_l1 += float(np.abs(p-before).sum())
            self.correction_count += 1
        return p

    def to_dict(self):
        return asdict(self)
