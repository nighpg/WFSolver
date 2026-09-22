"""Declarative YAML/JSON CLI with stable error exit codes."""
import argparse
from dataclasses import fields
import json
import logging
from pathlib import Path
import sys

from . import (ArrayInitialCondition, BetaInitialCondition, DeltaInitialCondition,
               DensityInitialCondition, ForwardSolver, PiecewiseConstant, SimulationResult,
               SolverConfig, UniformGrid, WrightFisherModel)
from .coefficients import PARAMETERS
from .exceptions import ConfigurationError, NumericalError
from .io import dumps
from .operator import build_generator


def _mapping(value, name, allowed):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ConfigurationError(f"{name} must be a mapping with string keys")
    unknown = set(value)-set(allowed)
    if unknown:
        raise ConfigurationError(f"unknown {name} fields: {', '.join(sorted(unknown))}")
    return dict(value)


def from_config(data):
    """Reconstruct a declarative model, including normalized saved configs."""
    data = _mapping(data, "configuration", ("schema_version", "model", "grid", "solver", "initial_condition", "output"))
    if type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1:
        raise ConfigurationError("unsupported configuration schema_version")
    model_config = _mapping(data.get("model", {}), "model", (f.name for f in fields(WrightFisherModel)))
    for name in PARAMETERS:
        value = model_config.get(name)
        if isinstance(value, dict):
            value = _mapping(value, name, ("type", "breakpoints", "values"))
            if value.get("type") != "piecewise_constant":
                raise ConfigurationError("CLI supports only declarative piecewise_constant schedules")
            try:
                model_config[name] = PiecewiseConstant(value["breakpoints"], value["values"])
            except KeyError as exc:
                raise ConfigurationError("schedule requires breakpoints and values") from exc
    if model_config.get("drift_override") is not None:
        raise ConfigurationError("callable models cannot be reconstructed by the CLI")
    model = WrightFisherModel(**model_config)
    grid_config = _mapping(data.get("grid", {}), "grid", ("type", "points"))
    if grid_config.pop("type", "uniform") != "uniform":
        raise ConfigurationError("v1 supports uniform grids only")
    grid = UniformGrid(**grid_config)
    solver_config = _mapping(data.get("solver", {}), "solver", (f.name for f in fields(SolverConfig)))
    solver = ForwardSolver(model, grid, SolverConfig(**solver_config))
    initial_config = _mapping(data.get("initial_condition", {}), "initial_condition",
                              ("type", "x0", "values", "normalize", "alpha", "beta", "projection"))
    kind = initial_config.pop("type", None)
    projection = initial_config.pop("projection", "linear_hat")
    if projection != "linear_hat":
        raise ConfigurationError("only linear_hat density projection is supported")
    if "normalize" in initial_config and not isinstance(initial_config["normalize"], bool):
        raise ConfigurationError("normalize must be a boolean")
    classes = {"delta": DeltaInitialCondition, "array": ArrayInitialCondition,
               "density": DensityInitialCondition, "beta": BetaInitialCondition}
    if not isinstance(kind, str) or kind not in classes:
        raise ConfigurationError("initial_condition.type must be delta, array, density, or beta")
    if isinstance(initial_config.get("values"), dict):
        raise ConfigurationError("callable initial conditions cannot be reconstructed by the CLI")
    try:
        initial = classes[kind](**initial_config)
    except TypeError as exc:
        raise ConfigurationError(f"invalid {kind} initial condition: {exc}") from exc
    output = _mapping(data.get("output", {}), "output", ("format", "include_full_distribution", "summary_csv"))
    if output.get("format", "npz") != "npz" or output.get("include_full_distribution", True) is not True:
        raise ConfigurationError("v1 saves full NPZ distributions; reduce output times to save memory")
    if output.get("summary_csv") is not None and not isinstance(output["summary_csv"], str):
        raise ConfigurationError("summary_csv must be a path string")
    return solver, initial, output


def read_config(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ConfigurationError("configuration must be UTF-8 text") from exc
    if Path(path).suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigurationError(f"invalid JSON: {exc}") from exc
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError("YAML requires pip install 'wf-forward[cli]'") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML: {exc}") from exc
    return from_config(data)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wf-forward")
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"), default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="solve a YAML/JSON configuration")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="validate without time integration")
    validate.add_argument("config")
    inspect = sub.add_parser("inspect", help="inspect NPZ metadata and summary")
    inspect.add_argument("result")
    inspect.add_argument("--blas", action="store_true")
    plot = sub.add_parser("plot", help="plot internal density and endpoint probabilities")
    plot.add_argument("result")
    plot.add_argument("--output", required=True)
    serve = sub.add_parser("serve", help="open the local simulation Web UI")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", dest="open_browser")
    sfs = sub.add_parser("sfs", help="calculate neutral demographic expected-site-count spectra from JSON")
    sfs.add_argument("config")
    sfs.add_argument("--output", required=True)
    sfs.add_argument("--csv")
    sample = sub.add_parser("sample", help="sample seeded CTMC trajectories from a saved NPZ result")
    sample.add_argument("result")
    sample.add_argument("--paths", type=int, default=32)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--max-events", type=int, default=5_000_000)
    sample.add_argument("--output", required=True)
    sample.add_argument("--csv")
    benchmark = sub.add_parser("benchmark", help="run isolated-process benchmarks")
    benchmark.add_argument("--grid-points", default="1001,10001,100001")
    benchmark.add_argument("--methods", default="implicit_euler")
    benchmark.add_argument("--cases", default="neutral,selection,mutation")
    benchmark.add_argument("--output-counts", default="10")
    benchmark.add_argument("--duration", type=float, default=1.0)
    benchmark.add_argument("--dt", type=float, default=0.1)
    benchmark.add_argument("--timeout", type=float, default=120)
    benchmark.add_argument("--output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    try:
        if args.command in ("run", "validate"):
            solver, initial, output = read_config(args.config)
            if args.command == "validate":
                initial.to_mass(solver.grid)
                for t in sorted({0.0, *solver.model.all_breakpoints()}):
                    build_generator(solver.model, solver.grid, t)
                required = len(solver.config.output_times)*solver.grid.points*8
                if required > solver.config.max_output_bytes:
                    raise ConfigurationError(f"output requires {required} bytes, above max_output_bytes")
                print(dumps({"valid": True, "output_bytes": required, "model": solver.model.to_dict()}))
            else:
                result = solver.solve(initial)
                result.save_npz(args.output)
                if output.get("summary_csv"):
                    result.to_summary_csv(output["summary_csv"])
                logging.info("saved %s", args.output)
        elif args.command == "inspect":
            result = SimulationResult.load_npz(args.result)
            print(dumps({"metadata": result.metadata, "diagnostics": result.diagnostics,
                         "shape": result.probability_mass.shape, "final_mean": float(result.mean()[-1]),
                         "final_p_at_zero": float(result.p_at_zero[-1]),
                         "final_p_at_one": float(result.p_at_one[-1])}))
            if args.blas:
                import numpy as np
                np.show_config()
        elif args.command == "plot":
            from .plotting import plot_result
            plot_result(SimulationResult.load_npz(args.result), args.output)
        elif args.command == "serve":
            from .web import serve
            serve(args.port, open_browser=args.open_browser)
        elif args.command == "sfs":
            from .sfs import calculate_sfs
            import json
            from pathlib import Path
            try:
                config = json.loads(Path(args.config).read_text())
            except (ValueError, UnicodeError) as exc:
                raise ConfigurationError("Invalid SFS JSON configuration") from exc
            result = calculate_sfs(config)
            result.save_npz(args.output)
            if args.csv:
                result.to_csv(args.csv)
            logging.info("saved SFS to %s", args.output)
        elif args.command == "sample":
            result = SimulationResult.load_npz(args.result)
            samples = result.sample_trajectories(args.paths, args.seed, args.max_events)
            samples.save_npz(args.output)
            if args.csv:
                samples.to_csv(args.csv)
            logging.info("saved %d trajectories to %s", args.paths, args.output)
        else:
            from .benchmark import run_benchmarks
            run_benchmarks(args)
        return 0
    except ConfigurationError as exc:
        logging.error("configuration/input: %s", exc)
        return 2
    except NumericalError as exc:
        logging.error("numerical: %s", exc)
        return 3
    except OSError as exc:
        logging.error("I/O: %s", exc)
        return 4
    except Exception:
        logging.exception("internal error")
        return 5
