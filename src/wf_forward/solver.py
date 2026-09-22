"""Public solver; diploid Wright–Fisher diffusion in generations."""
from dataclasses import asdict
from datetime import datetime, timezone
import logging
import platform
from pathlib import Path
import subprocess
from time import perf_counter
import warnings

import numpy as np
import scipy

from .config import SolverConfig
from .diagnostics import ProbabilityDiagnostics
from .exceptions import ConfigurationError, ModelConsistencyWarning, NumericalError, ResolutionWarning
from .grid import ArrayInitialCondition, DeltaInitialCondition, UniformGrid
from .integrators import OperatorProvider, integrate_expm, integrate_implicit
from .result import SimulationResult

logger = logging.getLogger(__name__)
VERSION = "0.1.0"


def environment_metadata():
    source_root = Path(__file__).resolve().parents[2]
    try:
        if not (source_root/".git").exists():
            raise FileNotFoundError("source checkout has no Git metadata")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=source_root, timeout=2, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"package_version": VERSION, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "macos": platform.mac_ver()[0], "architecture": platform.machine(), "git_commit": commit}


class ForwardSolver:
    def __init__(self, model, grid=None, config=None):
        self.model = model
        self.grid = grid if grid is not None else UniformGrid()
        self.config = config if config is not None else SolverConfig()
        if not isinstance(self.grid, UniformGrid) or not isinstance(self.config, SolverConfig):
            raise ConfigurationError("expected UniformGrid and SolverConfig")
        if self.config.method == "expm" and not model.is_time_homogeneous():
            raise ConfigurationError("expm requires a time-homogeneous model; use implicit_euler or auto")

    def solve(self, initial):
        """Evolve mass from t=0. Full output costs 8*T*grid.points bytes."""
        started, clock = datetime.now(timezone.utc).isoformat(), perf_counter()
        config, model, grid = self.config, self.model, self.grid
        required_bytes = len(config.output_times)*grid.points*8
        if required_bytes > config.max_output_bytes:
            raise ConfigurationError(f"full output requires {required_bytes} bytes, above max_output_bytes; "
                                     "reduce output times or explicitly raise the limit")
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            warnings.warn("x86_64 macOS execution; use native arm64 for Apple Silicon benchmarks",
                          ModelConsistencyWarning)
        if isinstance(initial, (list, tuple, np.ndarray)):
            initial = ArrayInitialCondition(initial)
        if not hasattr(initial, "to_mass") or not hasattr(initial, "to_dict"):
            raise ConfigurationError("initial must be an initial-condition object or mass array")
        p = initial.to_mass(grid)
        if isinstance(initial, DeltaInitialCondition) and 0 < initial.x0 < 1:
            near_endpoint = min(initial.x0, 1-initial.x0)
            if grid.dx > near_endpoint/2:
                warnings.warn("delta near an endpoint is under-resolved; use dx <= min(x0,1-x0)/2",
                              ResolutionWarning)
        initial_endpoints = [float(p[0]), float(p[-1])]
        diagnostics = ProbabilityDiagnostics()
        diagnostics.check(p, config)
        provider = OperatorProvider(model, grid)
        provider.get(0)  # Validate endpoint drift even for a t=0-only run.
        method = config.method
        if method == "auto":
            method = "expm" if model.is_time_homogeneous() else "implicit_euler"
        logger.info("model=%s points=%s dx=%g method=%s times=[%g,%g]", model.to_dict(),
                    grid.points, grid.dx, method, config.output_times[0], config.output_times[-1])
        integrator = integrate_expm if method == "expm" else integrate_implicit
        t0 = perf_counter()
        try:
            P, integration = integrator(p, np.asarray(config.output_times), provider, diagnostics, config)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise NumericalError(f"integration failed: {exc}") from exc
        integration["integration_seconds"] = perf_counter()-t0
        integration["operator_build_seconds"] = provider.build_seconds
        integration["operator_builds"] = provider.build_count
        integration.update(provider.timings)
        normalized = {"schema_version": 1, "model": model.to_dict(),
                      "grid": {"type": "uniform", "points": grid.points},
                      "initial_condition": initial.to_dict(), "solver": asdict(config)}
        metadata = {"schema_version": 1, "config": normalized,
                    "environment": environment_metadata(), "started_at": started,
                    "elapsed_seconds": perf_counter()-clock, "method": method,
                    "coefficient_time_unit": "generation", "time_unit": "generation",
                    "endpoint_semantics": "occupancy" if model.boundary == "mutation" else "absorption",
                    "reconstructible": not any(callable(getattr(model, n)) and
                        not hasattr(getattr(model, n), "to_dict") for n in
                        ("Ne", "selection", "dominance", "mutation_forward", "mutation_backward", "drift_override"))
                        and not callable(getattr(initial, "values", None)),
                    "integration": integration}
        report = diagnostics.to_dict() | {"max_peclet": provider.max_peclet,
                                         "initial_endpoint_mass": initial_endpoints,
                                         "output_bytes": required_bytes}
        result = SimulationResult(np.asarray(config.output_times), grid.x, P, metadata, report)
        logger.info("completed: seconds=%.4g steps=%s max_probability_error=%.3g minimum=%.3g max_Pe=%.3g",
                    metadata["elapsed_seconds"], integration["steps"], report["max_probability_error"],
                    report["minimum_probability"], report["max_peclet"])
        return result
