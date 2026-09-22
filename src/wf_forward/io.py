"""Atomic NPZ/CSV persistence. Loading never executes pickled Python objects."""
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
from collections.abc import Mapping

import numpy as np

from .exceptions import ConfigurationError

SCHEMA_VERSION = 1


def json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def dumps(value):
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, sort_keys=True)


@contextmanager
def atomic_file(path, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False, **kwargs) as stream:
        temp = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def save_npz(result, path):
    with atomic_file(path, binary=True) as stream:
        np.savez_compressed(stream, times=result.times, x=result.x,
                            probability_mass=result.probability_mass,
                            metadata_json=np.asarray(dumps(result.metadata)),
                            diagnostics_json=np.asarray(dumps(result.diagnostics)))


def load_npz(path):
    from .result import SimulationResult
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            diagnostics = json.loads(str(archive["diagnostics_json"].item()))
            if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
                    or metadata.get("schema_version") != SCHEMA_VERSION):
                raise ConfigurationError("unsupported NPZ schema_version")
            times, x, mass = (archive[n] for n in ("times", "x", "probability_mass"))
        result = SimulationResult(times, x, mass, metadata, diagnostics)
        tolerance = metadata.get("config", {}).get("solver", {}).get("probability_tolerance", 1e-10)
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ConfigurationError("invalid saved probability tolerance")
        if np.max(np.abs(mass.sum(axis=1)-1)) > tolerance:
            raise ConfigurationError("saved probability mass violates normalization")
        return result
    except (KeyError, TypeError, ValueError, AttributeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid NPZ result: {exc}") from exc


def _mutation_mode(result):
    return result.metadata.get("config", {}).get("model", {}).get("boundary") == "mutation"


def save_summary_csv(result, path):
    # Mutation output deliberately uses occupancy names, never fixation labels.
    endpoints = ["p_at_zero", "p_at_one"] if _mutation_mode(result) else ["p_loss", "p_fix"]
    columns = ["time", *endpoints, "p_segregating", "mean", "variance", "heterozygosity", "total_probability"]
    data = (result.times, result.p_at_zero, result.p_at_one, result.p_segregating,
            result.mean(), result.variance(), result.heterozygosity(), result.probability_mass.sum(axis=1))
    with atomic_file(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*data))


def save_distribution_csv(result, path, time_index=-1):
    p = result.probability_mass[time_index]
    time = result.times[time_index]
    mutation = _mutation_mode(result)
    with atomic_file(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "x", "probability_mass", "density", "state"))
        for i, (x, mass) in enumerate(zip(result.x, p)):
            if i == 0:
                density, state = "", "at_zero" if mutation else "loss"
            elif i == len(p)-1:
                density, state = "", "at_one" if mutation else "fixation"
            else:
                density, state = mass*(len(p)-1), "interior"
            writer.writerow((time, x, mass, density, state))
