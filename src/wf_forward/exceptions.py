"""Public errors and warnings."""


class ConfigurationError(ValueError):
    """An invalid model, grid, initial condition, or configuration."""


class NumericalError(RuntimeError):
    """A failed solve or probability invariant."""


class ResolutionWarning(UserWarning):
    """The spatial or temporal resolution may be insufficient."""


class ModelConsistencyWarning(UserWarning):
    """A boundary or coefficient convention deserves attention."""
