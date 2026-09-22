"""Headless plots keep endpoint occupancy separate from interior density."""
from pathlib import Path

import numpy as np

from .exceptions import ConfigurationError


def plot_result(result, path):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ConfigurationError("plotting requires pip install 'wf-forward[plot]'") from exc
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    density, endpoints = fig.subplots(1, 2)
    initial = result.metadata.get("config", {}).get("initial_condition", {})
    skip_delta = len(result.times) > 1 and result.times[0] == 0 and initial.get("type") == "delta"
    first = 1 if skip_delta else 0
    indices = np.unique(np.linspace(first, len(result.times)-1, min(5, len(result.times)-first)).astype(int))
    if skip_delta:
        density.axvline(initial["x0"], color="0.5", linestyle=":", label=f"Initial delta x={initial['x0']:g}")
    for i in indices:
        density.plot(result.x[1:-1], result.probability_mass[i, 1:-1]*(len(result.x)-1),
                     label=f"t={result.times[i]:g}")
    density.set(xlabel="Allele frequency x", ylabel="Interior density", title="Wright–Fisher distribution")
    density.legend()
    mutation = result.metadata.get("endpoint_semantics") == "occupancy"
    endpoints.plot(result.times, result.p_at_zero, label="At x=0" if mutation else "Loss")
    endpoints.plot(result.times, result.p_at_one, label="At x=1" if mutation else "Fixation")
    endpoints.plot(result.times, result.p_segregating, label="Interior")
    endpoints.set(xlabel="Time (generations)", ylabel="Probability", ylim=(-0.02, 1.02),
                  title="Grid endpoint occupancy" if mutation else "Absorption probabilities")
    endpoints.legend()
    for ax in (density, endpoints):
        ax.grid(alpha=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    return fig
