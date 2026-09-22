"""Isolated workers report per-process peak RSS and separated timings."""
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from . import ForwardSolver, WrightFisherModel, UniformGrid, SolverConfig, DeltaInitialCondition
from .config import positive_float, positive_integer
from .exceptions import ConfigurationError
from .io import atomic_file, dumps


def worker(case):
    model_args = {"Ne": 10_000}
    if case["case"] == "selection":
        model_args["selection"] = 0.1
    elif case["case"] == "mutation":
        model_args.update(mutation_forward=1e-4, mutation_backward=2e-4, boundary="mutation")
    start = perf_counter()
    result = ForwardSolver(WrightFisherModel(**model_args), UniformGrid(case["points"]),
                           SolverConfig(method=case["method"], dt=case["dt"],
                                        output_times=np.linspace(0, case["duration"], case["outputs"]),
                                        max_output_bytes=2*1024**3)).solve(DeltaInitialCondition(0.3))
    solve_seconds = perf_counter()-start
    with tempfile.TemporaryDirectory() as directory:
        start = perf_counter()
        result.save_npz(os.path.join(directory, "result.npz"))
        save_seconds = perf_counter()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss*1024
    return case | {"status": "ok", "solve_seconds": solve_seconds, "save_seconds": save_seconds,
                   "peak_rss_bytes": rss_bytes, "environment": result.metadata["environment"],
                   "integration": result.metadata["integration"], "diagnostics": result.diagnostics}


def run_benchmarks(args):
    try:
        points = [int(v) for v in args.grid_points.split(",")]
        counts = [int(v) for v in args.output_counts.split(",")]
    except ValueError as exc:
        raise ConfigurationError("benchmark sizes must be comma-separated integers") from exc
    for value in points:
        positive_integer(value, "grid points", 101)
    for value in counts:
        positive_integer(value, "output counts", 2)
    for name in ("duration", "dt", "timeout"):
        positive_float(getattr(args, name), name)
    methods, cases = args.methods.split(","), args.cases.split(",")
    if set(methods)-{"expm", "implicit_euler"} or set(cases)-{"neutral", "selection", "mutation"}:
        raise ConfigurationError("invalid benchmark method or case")
    records = []
    for n, method, case, count in itertools.product(points, methods, cases, counts):
        settings = {"points": n, "method": method, "case": case, "outputs": count,
                    "duration": args.duration, "dt": args.dt}
        try:
            child = subprocess.run([sys.executable, "-m", "wf_forward.benchmark", json.dumps(settings)],
                                   capture_output=True, text=True, timeout=args.timeout)
            if child.returncode != 0:
                record = settings | {"status": "failed", "error": child.stderr[-2000:]}
            else:
                record = json.loads(child.stdout)
        except subprocess.TimeoutExpired:
            record = settings | {"status": "timeout", "timeout_seconds": args.timeout}
        records.append(record)
        print(dumps(record), flush=True)
    if args.output:
        with atomic_file(args.output) as stream:
            stream.write(dumps(records)+"\n")
    if any(record["status"] != "ok" for record in records):
        from .exceptions import NumericalError
        raise NumericalError("some benchmark cases failed or timed out; see recorded results")


if __name__ == "__main__":
    print(dumps(worker(json.loads(sys.argv[1]))))
