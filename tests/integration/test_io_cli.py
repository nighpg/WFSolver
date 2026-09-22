import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wf_forward import *
from wf_forward.cli import from_config, main


def configuration():
    return {"schema_version": 1, "model": {"Ne": 100}, "grid": {"points": 101},
            "initial_condition": {"type": "delta", "x0": .3},
            "solver": {"output_times": [0, 1, 2]}}


def test_npz_roundtrip_and_reconstruct(tmp_path):
    solver, initial, _ = from_config(configuration())
    result = solver.solve(initial)
    path = tmp_path/"nested"/"result.npz"
    result.save_npz(path)
    loaded = SimulationResult.load_npz(path)
    np.testing.assert_array_equal(result.probability_mass, loaded.probability_mass)
    assert loaded.metadata["environment"] == result.metadata["environment"]
    assert loaded.diagnostics == result.diagnostics
    solver2, initial2, _ = from_config(loaded.metadata["config"])
    np.testing.assert_array_equal(solver2.solve(initial2).probability_mass, result.probability_mass)
    assert not loaded.probability_mass.flags.writeable


def test_schedule_roundtrip(tmp_path):
    data = configuration()
    data["model"]["Ne"] = {"type": "piecewise_constant", "breakpoints": [0, 1], "values": [100, 20]}
    data["solver"]["dt"] = .25
    solver, initial, _ = from_config(data)
    result = solver.solve(initial)
    path = tmp_path/"result.npz"
    result.save_npz(path)
    config = SimulationResult.load_npz(path).metadata["config"]
    replay, initial, _ = from_config(config)
    np.testing.assert_array_equal(replay.solve(initial).probability_mass, result.probability_mass)


def test_cli_matches_api_yaml_and_error_codes(tmp_path, capsys):
    import yaml
    data = configuration()
    data["output"] = {"summary_csv": str(tmp_path/"summary.csv")}
    config_path = tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    path = tmp_path/"result.npz"
    assert main(["validate", str(config_path)]) == 0
    assert main(["run", str(config_path), "--output", str(path)]) == 0
    expected = from_config(data)[0].solve(DeltaInitialCondition(.3))
    np.testing.assert_array_equal(SimulationResult.load_npz(path).probability_mass, expected.probability_mass)
    assert main(["inspect", str(path)]) == 0
    assert (tmp_path/"summary.csv").exists()
    config_path.write_text("schema_version: 100\n")
    assert main(["validate", str(config_path)]) == 2
    assert main(["inspect", str(tmp_path/"missing.npz")]) == 4
    data["solver"] = {"method": "implicit_euler", "dt": .1, "max_steps": 1, "output_times": [0, 1]}
    config_path.write_text(yaml.safe_dump(data))
    assert main(["run", str(config_path), "--output", str(path)]) == 3


@pytest.mark.parametrize("change", [
    {"typo": 1}, {"model": {"mutaton_forward": .01}}, {"solver": {"output_times": [1, 0]}},
    {"initial_condition": {"type": "delta", "x0": .3, "alpha": 1}},
    {"model": {"Ne": {"type": "callable", "values": [1], "breakpoints": [0]}}},
    {"initial_condition": {"type": "array", "values": [1], "normalize": "false"}},
    {"schema_version": True}, {"output": {"include_full_distribution": False}},
    {"schema_version": 1.0}, {"initial_condition": {"type": []}},
])
def test_unknown_or_unsafe_configuration(change):
    with pytest.raises(ConfigurationError):
        from_config(configuration() | change)


def test_unknown_and_corrupt_npz(tmp_path):
    path = tmp_path/"bad.npz"
    np.savez(path, metadata_json=json.dumps({"schema_version": 99}), diagnostics_json="{}")
    with pytest.raises(ConfigurationError):
        SimulationResult.load_npz(path)
    path.write_bytes(b"not an npz")
    with pytest.raises(ConfigurationError):
        SimulationResult.load_npz(path)


def test_non_utf8_configuration_is_input_error(tmp_path):
    path = tmp_path/"bad.yaml"
    path.write_bytes(b"\xff\xfe")
    assert main(["validate", str(path)]) == 2


def test_mutation_csv_endpoint_names(tmp_path):
    model = WrightFisherModel(Ne=100, mutation_forward=.001, boundary="mutation")
    result = ForwardSolver(model, UniformGrid(101), SolverConfig(output_times=[0, 1])).solve(DeltaInitialCondition(.3))
    result.to_summary_csv(tmp_path/"summary.csv")
    result.to_distribution_csv(tmp_path/"distribution.csv")
    with (tmp_path/"summary.csv").open() as f:
        columns = next(csv.reader(f))
    assert "p_at_one" in columns and "p_fix" not in columns
    with (tmp_path/"distribution.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["density"] == rows[-1]["density"] == ""
    assert rows[0]["state"] == "at_zero" and rows[-1]["state"] == "at_one"


def test_module_cli_and_plot(tmp_path):
    data = configuration()
    path = tmp_path/"config.json"
    path.write_text(json.dumps(data))
    env = os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[2]/"src"),
                        "MPLCONFIGDIR": str(tmp_path/"mpl")}
    run = subprocess.run([sys.executable, "-m", "wf_forward", "run", str(path),
                          "--output", str(tmp_path/"result.npz")], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    plot = subprocess.run([sys.executable, "-m", "wf_forward", "plot", str(tmp_path/"result.npz"),
                           "--output", str(tmp_path/"plot.png")], env=env, capture_output=True, text=True)
    assert plot.returncode == 0, plot.stderr
    assert (tmp_path/"plot.png").read_bytes().startswith(b"\x89PNG")
