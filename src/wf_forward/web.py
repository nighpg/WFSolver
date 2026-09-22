"""Loopback-only Web UI, with one cancellable simulation worker at a time.

No third-party web dependencies or external assets are used. Results live in a
temporary directory until server shutdown; download them to retain them.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import logging
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit, parse_qs
import warnings
import webbrowser

import numpy as np

from .cli import from_config
from .config import positive_integer
from .exceptions import ConfigurationError, NumericalError
from .io import atomic_file, dumps
from .result import SimulationResult
from .sfs import calculate_sfs, validate_sfs_config

MAX_POINTS = 20_001
MAX_FRAMES = 201
MAX_BODY = 128 * 1024
MAX_STEPS = 200_000
MAX_BYTES = 64 * 1024**2
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


def validate_web_config(data):
    """Apply web resource limits before allocating any user-sized grid."""
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a JSON object.")
    data = json.loads(dumps(data))
    if data.get("output"):
        raise ConfigurationError("Download results instead of specifying output paths in the Web UI.")
    grid = data.setdefault("grid", {})
    solver = data.setdefault("solver", {})
    if not isinstance(grid, dict) or not isinstance(solver, dict):
        raise ConfigurationError("grid / solver must be mappings")
    points = positive_integer(grid.get("points", 2001), "points", 101)
    times = solver.get("output_times", [0, 100, 1000])
    if points > MAX_POINTS or not isinstance(times, list) or not 1 <= len(times) <= MAX_FRAMES:
        raise ConfigurationError(f"The Web UI supports up to {MAX_POINTS:,} grid points and {MAX_FRAMES} output times.")
    for key, limit in (("max_steps", MAX_STEPS), ("max_output_bytes", MAX_BYTES)):
        solver.setdefault(key, limit)
        if positive_integer(solver[key], key) > limit:
            raise ConfigurationError(f"Web UI requires {key} <= {limit}")
    # Shared validation keeps API, CLI and browser numerical conventions equal.
    from_config(data)
    return data


def result_summary(result):
    return {"times": result.times, "mean": result.mean(), "variance": result.variance(),
            "heterozygosity": result.heterozygosity(), "p_at_zero": result.p_at_zero,
            "p_at_one": result.p_at_one, "p_interior": result.p_segregating,
            "total_probability": result.probability_mass.sum(axis=1),
            "metadata": result.metadata, "diagnostics": result.diagnostics}


def density_frame(result, index, max_bins=801):
    """Group adjacent interior nodal cells, preserving integrated probability."""
    if not 0 <= index < len(result.times):
        raise ConfigurationError("Output time index is out of range.")
    mass = result.probability_mass[index, 1:-1]
    groups = np.array_split(np.arange(len(mass)), min(len(mass), max_bins))
    dx = 1/(len(result.x)-1)
    centers, density, widths = [], [], []
    for group in groups:
        centers.append(float(result.x[1+group[0]]+result.x[1+group[-1]])/2)
        widths.append(len(group)*dx)
        density.append(float(mass[group].sum())/widths[-1])
    return {"index": index, "time": float(result.times[index]), "x": centers,
            "density": density, "widths": widths, "aggregated": len(mass) > max_bins}


def worker(directory):
    """Private subprocess entrypoint; only the server chooses the directory."""
    root = Path(directory)
    try:
        config = json.loads((root/"config.json").read_text())
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("default")
            solver, initial, _ = from_config(config)
            result = solver.solve(initial)
        result.save_npz(root/"result.npz")
        result.to_summary_csv(root/"summary.csv")
        # Export effective settings, including the actual web resource limits.
        with atomic_file(root/"config.json") as stream:
            stream.write(dumps(result.metadata["config"]))
        summary = result_summary(result)
        summary["warnings"] = list(dict.fromkeys(str(w.message) for w in captured))[:20]
        with atomic_file(root/"summary.json") as stream:
            stream.write(dumps(summary))
        outcome = {"status": "complete"}
    except (ConfigurationError, NumericalError) as exc:
        outcome = {"status": "failed", "error": str(exc)}
    except Exception:
        logging.exception("Web simulation worker failed")
        outcome = {"status": "failed", "error": "Calculation failed. Check the server log."}
    with atomic_file(root/"outcome.json") as stream:
        stream.write(dumps(outcome))


def sampling_worker(directory):
    root = Path(directory)
    try:
        options = json.loads((root/"options.json").read_text())
        result = SimulationResult.load_npz(root/"source.npz")
        samples = result.sample_trajectories(**options)
        samples.save_npz(root/"trajectories.npz")
        samples.to_csv(root/"trajectories.csv")
        with atomic_file(root/"trajectories.json") as stream:
            stream.write(dumps({"times": samples.times, "frequencies": samples.frequencies,
                                "metadata": samples.metadata}))
        outcome = {"status": "complete"}
    except (ConfigurationError, NumericalError) as exc:
        outcome = {"status": "failed", "error": str(exc)}
    except Exception:
        logging.exception("Trajectory sampling failed")
        outcome = {"status": "failed", "error": "Trajectory sampling failed. Check the server log."}
    finally:
        (root/"source.npz").unlink(missing_ok=True)
    with atomic_file(root/"outcome.json") as stream:
        stream.write(dumps(outcome))


def sfs_worker(directory):
    root = Path(directory)
    try:
        result = calculate_sfs(json.loads((root/'config.json').read_text()))
        result.save_npz(root/'sfs.npz')
        result.to_csv(root/'sfs.csv')
        with atomic_file(root/'sfs.json') as stream:
            stream.write(dumps(result.to_dict()))
        outcome = {'status': 'complete'}
    except (ConfigurationError, NumericalError) as exc:
        outcome = {'status': 'failed', 'error': str(exc)}
    except Exception:
        logging.exception('SFS worker failed')
        outcome = {'status': 'failed', 'error': 'SFS calculation failed. Check the server log.'}
    with atomic_file(root/'outcome.json') as stream:
        stream.write(dumps(outcome))


@dataclass
class Job:
    id: str
    directory: Path
    process: subprocess.Popen
    created: float = field(default_factory=time.monotonic)
    status: str = "running"
    error: str | None = None
    elapsed: float | None = None
    result: SimulationResult | None = None
    sampling: "Job | None" = None
    last_sampling: "Job | None" = None
    parent_id: str | None = None
    kind: str = "simulation"

    def info(self):
        return {"id": self.id, "status": self.status, "error": self.error, "kind": self.kind,
                "elapsed": self.elapsed if self.elapsed is not None else time.monotonic()-self.created}


class JobManager:
    def __init__(self, timeout=120, keep=3):
        self.temp = tempfile.TemporaryDirectory(prefix="wf-forward-web-")
        self.jobs = OrderedDict()
        self.lock = threading.RLock()
        self.timeout, self.keep = timeout, keep
        self.closed = False
        self.watchers = []

    def busy(self):
        return self.closed or any(j.status == "running" or (j.sampling and j.sampling.status == "running")
                                  for j in self.jobs.values())

    def start(self, config, kind="simulation"):
        if kind not in ("simulation", "sfs"):
            raise ConfigurationError("Unknown job kind")
        config = validate_sfs_config(config) if kind == "sfs" else validate_web_config(config)
        with self.lock:
            if self.busy():
                raise RuntimeError("Another calculation is running. Wait for it to finish or cancel it.")
            matching = [j for j in self.jobs.values() if j.kind == kind]
            while len(matching) >= self.keep:
                old = matching.pop(0)
                del self.jobs[old.id]
                shutil.rmtree(old.directory, ignore_errors=True)
            job_id = secrets.token_hex(12)
            root = Path(self.temp.name)/job_id
            root.mkdir()
            (root/"config.json").write_text(dumps(config), encoding="utf-8")
            with (root/"worker.log").open("wb") as log:
                process = subprocess.Popen([sys.executable, "-m", "wf_forward.web", "--sfs-worker" if kind == "sfs" else "--worker", str(root)],
                                           stdout=log, stderr=log)
            job = Job(job_id, root, process, kind=kind)
            self.jobs[job_id] = job
            watch = threading.Thread(target=self._watch, args=(job,), daemon=True)
            self.watchers.append(watch)
            watch.start()
            return job.info()

    def start_sampling(self, job_id, options):
        if not isinstance(options, dict) or set(options)-{"paths", "seed"}:
            raise ConfigurationError("sampling options must contain only paths and seed")
        paths = positive_integer(options.get("paths", 32), "paths")
        seed = positive_integer(options.get("seed", 42), "seed", 0)
        if paths > 128 or seed > 2**32-1:
            raise ConfigurationError("Web sampling supports 1–128 paths and a seed from 0 to 4294967295")
        with self.lock:
            parent = self.get(job_id)
            if parent.status != "complete" or parent.kind != "simulation":
                raise ConfigurationError("Complete a forward simulation before sampling trajectories.")
            if self.busy():
                raise RuntimeError("Another calculation is running. Wait for it to finish or cancel it.")
            if parent.sampling and parent.sampling is not parent.last_sampling:
                shutil.rmtree(parent.sampling.directory, ignore_errors=True)
            sample_id = secrets.token_hex(12)
            root = parent.directory/("sampling-"+sample_id)
            root.mkdir()
            shutil.copyfile(parent.directory/"result.npz", root/"source.npz")
            (root/"options.json").write_text(dumps({"paths": paths, "seed": seed, "max_events": 5_000_000}))
            with (root/"worker.log").open("wb") as log:
                process = subprocess.Popen([sys.executable, "-m", "wf_forward.web", "--sample-worker", str(root)],
                                           stdout=log, stderr=log)
            child = Job(sample_id, root, process, parent_id=parent.id)
            parent.sampling = child
            watcher = threading.Thread(target=self._watch, args=(child,), daemon=True)
            self.watchers.append(watcher)
            watcher.start()
            return child.info()

    def cancel_sampling(self, job_id):
        with self.lock:
            sample = self.get(job_id).sampling
            if sample is None:
                return {"status": "idle"}
            if sample.status == "running":
                sample.status = "cancelled"
                self._stop(sample.process)
                sample.elapsed = time.monotonic()-sample.created
            return sample.info()

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _watch(self, job):
        try:
            job.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._stop(job.process)
            with self.lock:
                if job.status == "running":
                    job.status = "failed"
                    job.error = f"Calculation exceeded {self.timeout:g} seconds. Reduce the grid size, path count, or duration."
        with self.lock:
            if job.status == "running":
                try:
                    outcome = json.loads((job.directory/"outcome.json").read_text())
                    job.status, job.error = outcome["status"], outcome.get("error")
                except (OSError, ValueError, KeyError):
                    job.status, job.error = "failed", "The worker exited unexpectedly. Check the configuration and run again."
            job.elapsed = time.monotonic()-job.created
            if job.parent_id is not None and job.status == "complete":
                parent = self.get(job.parent_id)
                if parent.last_sampling:
                    shutil.rmtree(parent.last_sampling.directory, ignore_errors=True)
                parent.last_sampling = job

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Result not found. Run the simulation again.")
            return self.jobs[job_id]

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job.status == "running":
                job.status = "cancelled"
                self._stop(job.process)
                job.elapsed = time.monotonic()-job.created
            return job.info()

    def frame(self, job_id, index):
        with self.lock:
            job = self.get(job_id)
            if job.status != "complete":
                raise ConfigurationError("Results are not ready yet.")
            if job.result is None:
                job.result = SimulationResult.load_npz(job.directory/"result.npz")
            return density_frame(job.result, index)

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                self.cancel(job.id)
                self.cancel_sampling(job.id)
        for watcher in self.watchers:
            watcher.join(timeout=5)
        self.temp.cleanup()


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=8765, manager=None):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ConfigurationError("port must be an integer between 0 and 65535")
        super().__init__(("127.0.0.1", port), Handler)
        self.manager = manager or JobManager()
        self.token = secrets.token_urlsafe(32)
        actual = self.server_address[1]
        self.hosts = {f"127.0.0.1:{actual}", f"localhost:{actual}"}

    def server_close(self):
        super().server_close()
        if hasattr(self, "manager"):
            self.manager.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "WrightFisher/0.1"

    def log_message(self, format, *args):
        logging.debug(format, *args)

    def _send(self, status, body, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(body, bytes):
            body = dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Referrer-Policy", "same-origin")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _trusted(self, mutation=False):
        if self.headers.get("Host") not in self.server.hosts:
            self._send(403, {"error": "Local access only"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://"+host for host in self.server.hosts}:
            self._send(403, {"error": "Origin not allowed"})
            return False
        if mutation and not secrets.compare_digest(self.headers.get("X-WF-Token", ""), self.server.token):
            self._send(403, {"error": "Reload the page to reconnect."})
            return False
        return True

    def do_GET(self):
        if not self._trusted():
            return
        url = urlsplit(self.path)
        try:
            if url.path in STATIC:
                name, mime = STATIC[url.path]
                self._send(200, files("wf_forward").joinpath("web_static", name).read_bytes(), mime)
                return
            if url.path == "/api/session":
                with self.server.manager.lock:
                    latest = next((j for j in reversed(self.server.manager.jobs.values()) if j.kind == "simulation"), None)
                    latest_sfs = next((j for j in reversed(self.server.manager.jobs.values()) if j.kind == "sfs"), None)
                    self._send(200, {"token": self.server.token, "latest": latest.info() if latest else None,
                                     "latest_sfs": latest_sfs.info() if latest_sfs else None,
                                     "limits": {"points": MAX_POINTS, "frames": MAX_FRAMES,
                                                "timeout": self.server.manager.timeout}})
                return
            parts = url.path.strip("/").split("/")
            if len(parts) not in (3, 4) or parts[:2] != ["api", "jobs"]:
                self._send(404, {"error": "Not found"})
                return
            with self.server.manager.lock:
                job = self.server.manager.get(parts[2])
                if len(parts) == 3:
                    self._send(200, job.info())
                    return
                if job.status != "complete":
                    self._send(409, {"error": "Results are not ready yet."})
                    return
                action = parts[3]
                if job.kind == "sfs":
                    downloads = {"sfs.json": "application/json", "sfs.csv": "text/csv; charset=utf-8",
                                 "sfs.npz": "application/octet-stream", "config.json": "application/json"}
                    if action not in downloads:
                        self._send(404, {"error": "Not an SFS output"})
                        return
                    self._send(200, (job.directory/action).read_bytes(), downloads[action],
                               filename=action if action != "sfs.json" else None)
                    return
                if action == "sampling":
                    self._send(200, {"current": job.sampling.info() if job.sampling else None,
                                     "available": job.last_sampling is not None})
                    return
                if action in ("trajectories.json", "trajectories.csv", "trajectories.npz"):
                    if job.last_sampling is None:
                        self._send(409, {"error": "No sampled trajectories are available yet."})
                        return
                    mime = {"trajectories.json": "application/json", "trajectories.csv": "text/csv",
                            "trajectories.npz": "application/octet-stream"}[action]
                    self._send(200, (job.last_sampling.directory/action).read_bytes(), mime,
                               filename=action if action != "trajectories.json" else None)
                    return
                if action == "frame":
                    index = int(parse_qs(url.query).get("index", ["0"])[0])
                    self._send(200, self.server.manager.frame(job.id, index))
                    return
                downloads = {"result.npz": "application/octet-stream", "summary.csv": "text/csv; charset=utf-8",
                             "config.json": "application/json; charset=utf-8", "summary.json": "application/json; charset=utf-8"}
                if action not in downloads:
                    self._send(404, {"error": "Not found"})
                    return
                self._send(200, (job.directory/action).read_bytes(), downloads[action],
                           filename=action if action != "summary.json" else None)
        except KeyError as exc:
            self._send(404, {"error": str(exc)})
        except (ValueError, ConfigurationError) as exc:
            self._send(422, {"error": str(exc)})
        except OSError:
            self._send(500, {"error": "Could not read the results."})

    def do_POST(self):
        if not self._trusted(mutation=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self._send(413, {"error": "Configuration exceeds the size limit."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._send(415, {"error": "Content-Type must be application/json"})
                return
            def reject_constant(value):
                raise ValueError(f"Non-finite value {value}")
            data = json.loads(self.rfile.read(length), parse_constant=reject_constant)
            parts = urlsplit(self.path).path.strip("/").split("/")
            if parts == ["api", "jobs"]:
                self._send(202, self.server.manager.start(data))
            elif parts == ["api", "sfs"]:
                self._send(202, self.server.manager.start(data, kind="sfs"))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[-1] == "cancel":
                self._send(200, self.server.manager.cancel(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[-1] == "sample":
                self._send(202, self.server.manager.start_sampling(parts[2], data))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[-1] == "cancel-sampling":
                self._send(200, self.server.manager.cancel_sampling(parts[2]))
            else:
                self._send(404, {"error": "Not found"})
        except KeyError as exc:
            self._send(404, {"error": str(exc)})
        except (ValueError, ConfigurationError) as exc:
            self._send(422, {"error": str(exc)})
        except RuntimeError as exc:
            self._send(409, {"error": str(exc)})
        except OSError:
            logging.exception("Failed to start simulation")
            self._send(500, {"error": "Could not start the calculation."})


def serve(port=8765, open_browser=False):
    server = WebServer(port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Wright–Fisher Lab: {url}\nStop: Ctrl+C", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--sfs-worker":
        sfs_worker(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--sample-worker":
        sampling_worker(sys.argv[2])
    else:
        raise SystemExit("Start the Web UI with: wf-forward serve")
