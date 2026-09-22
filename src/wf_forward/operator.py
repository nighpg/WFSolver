"""O(G) upwind CTMC: Q[i,j] is the rate from i to j."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import diags

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class Generator:
    down: np.ndarray
    up: np.ndarray
    diagonal: np.ndarray
    max_peclet: float

    def as_sparse(self, format="csc"):
        return diags((self.down[1:], self.diagonal, self.up[:-1]), (-1, 0, 1), format=format)


def build_generator(model, grid, t=0.0, timings=None):
    x, dx = grid.x, grid.dx
    start = perf_counter()
    drift, variance = model.evaluate(x, t)
    evaluated = perf_counter()
    if drift[0] < 0 or drift[-1] > 0:
        raise ConfigurationError("drift points out of [0,1] at a boundary")
    if model.boundary == "absorbing" and (drift[0] != 0 or drift[-1] != 0):
        raise ConfigurationError("absorbing boundary requires zero endpoint drift")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        down = variance/(2*dx*dx) + np.maximum(-drift, 0)/dx
        up = variance/(2*dx*dx) + np.maximum(drift, 0)/dx
        diagonal = -(down+up)
    if not all(np.all(np.isfinite(a)) for a in (down, up, diagonal)):
        raise ConfigurationError("transition rates overflow; check Ne, drift, and grid resolution")
    D = variance[1:-1]/2
    numerator = np.abs(drift[1:-1])*dx
    with np.errstate(over="ignore", divide="ignore"):
        pe = np.divide(numerator, D, out=np.zeros_like(D), where=D > 0)
    pe[(D == 0) & (numerator > 0)] = np.inf
    if timings is not None:
        timings["coefficient_evaluation_seconds"] += evaluated-start
        timings["matrix_assembly_seconds"] += perf_counter()-evaluated
    return Generator(down, up, diagonal, float(np.max(pe)))
