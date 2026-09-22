"""Seeded, exact event-driven sampling of the solver's grid CTMC.

One-time marginal distributions do not specify temporal dependence. Rebuild Q
from the saved model and simulate its jumps, starting at the first saved time.
Coefficients must be constant or declarative piecewise-constant schedules.
"""
import csv
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import positive_integer
from .exceptions import ConfigurationError, NumericalError
from .grid import UniformGrid
from .io import atomic_file, dumps
from .operator import build_generator


@dataclass(frozen=True)
class TrajectoryResult:
    times: np.ndarray
    frequencies: np.ndarray  # (paths, saved times)
    metadata: dict

    def __post_init__(self):
        times = np.asarray(self.times, dtype=np.float64)
        values = np.asarray(self.frequencies, dtype=np.float64)
        if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
                or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ConfigurationError("invalid trajectory times")
        if (values.ndim != 2 or values.shape[1] != len(times) or not len(values)
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
            raise ConfigurationError("invalid trajectory frequencies")
        times.flags.writeable = values.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "frequencies", values)

    def save_npz(self, path):
        with atomic_file(path, binary=True) as stream:
            np.savez_compressed(stream, times=self.times, frequencies=self.frequencies,
                                metadata_json=np.asarray(dumps(self.metadata)))

    def to_csv(self, path):
        with atomic_file(path) as stream:
            writer = csv.writer(stream)
            writer.writerow(("path_id", "time", "allele_frequency"))
            for path_id, values in enumerate(self.frequencies):
                writer.writerows((path_id, t, x) for t, x in zip(self.times, values))


def sample_trajectories(result, paths=32, seed=42, max_events=5_000_000):
    """Return unconditioned CTMC paths at the result's saved observation times.

    Waiting times have mean 1/(q_down+q_up). At each jump, choose up with
    probability q_up/(q_down+q_up). Absorbing states persist; schedule changes
    can reactivate endpoints under mutation. No independent marginal draws or
    temporal interpolation are used to generate the samples.
    """
    started = perf_counter()
    paths = positive_integer(paths, "paths")
    seed = positive_integer(seed, "seed", 0)
    max_events = positive_integer(max_events, "max_events")
    if paths*len(result.times)*8 > 256*1024**2:
        raise ConfigurationError("trajectory output exceeds 256 MiB; reduce paths or output times")
    if not result.metadata.get("reconstructible", False):
        raise ConfigurationError("trajectory sampling requires a saved declarative model, not Python callables")
    from .cli import from_config
    solver, _, _ = from_config(result.metadata["config"])
    model = solver.model
    grid = UniformGrid(len(result.x))
    if not np.array_equal(grid.x, result.x):
        raise ConfigurationError("trajectory grid differs from the saved result")
    rng = np.random.Generator(np.random.PCG64(seed))
    # The saved mass has already passed solver/loader checks. Normalize only
    # this private categorical probability vector to remove roundoff drift.
    probability = result.probability_mass[0].copy()
    total = probability.sum()
    if abs(total-1) > solver.config.probability_tolerance:
        raise ConfigurationError("initial saved marginal does not sum to one")
    states = rng.choice(grid.points, size=paths, p=probability/total)
    frequencies = np.empty((paths, len(result.times)))
    frequencies[:, 0] = grid.x[states]
    left = float(result.times[0])
    events = np.unique(np.concatenate((result.times[1:], model.all_breakpoints())))
    events = events[(events > left) & (events <= result.times[-1])]
    output_index, jumps = 1, 0
    previous_key, q = None, None
    for target in events:
        key = model.operator_key(left)
        if key != previous_key:
            q = build_generator(model, grid, left)
            previous_key = key
        clocks = np.full(paths, left)
        active = np.arange(paths)
        while len(active):
            rates = -q.diagonal[states[active]]
            moving = rates > 0
            active, rates = active[moving], rates[moving]
            if not len(active):
                break
            with np.errstate(over="ignore", divide="ignore"):
                jump_time = clocks[active]+rng.exponential(scale=1/rates)
            if np.any(jump_time <= clocks[active]):
                raise NumericalError("trajectory waiting time is below floating-point time resolution")
            before_target = jump_time < target
            active, rates = active[before_target], rates[before_target]
            jump_time = jump_time[before_target]
            jumps += len(active)
            if jumps > max_events:
                raise NumericalError("trajectory event limit exceeded; reduce the path count, grid size, "
                                     "or simulation duration (no partial sample is returned)")
            if len(active):
                up = rng.random(len(active)) < q.up[states[active]]/rates
                states[active] += np.where(up, 1, -1)
                clocks[active] = jump_time
        if output_index < len(result.times) and target == result.times[output_index]:
            frequencies[:, output_index] = grid.x[states]
            output_index += 1
        left = float(target)
    metadata = {"schema_version": 1, "method": "exact_grid_ctmc", "rng": "PCG64", "seed": seed,
                "paths": paths, "events": jumps, "max_events": max_events,
                "elapsed_seconds": perf_counter()-started, "numpy_version": np.__version__,
                "start_time": float(result.times[0]), "time_unit": "generation",
                "source_config": result.metadata["config"], "source_method": result.metadata["method"],
                "source_started_at": result.metadata.get("started_at"),
                "observation_note": "Only saved times are retained; plotted connecting lines are not jump times."}
    return TrajectoryResult(result.times.copy(), frequencies, metadata)
