"""Tools for camera-coordinate background studies."""

from .camera import CameraImage
from .events import CameraEvents

__all__ = [
    "CameraEvents",
    "CameraImage",
    "plot_energy_slices",
    "plot_radial_acceptance",
]


def __getattr__(name: str):
    """Load Matplotlib-based helpers only when requested."""
    if name in {"plot_energy_slices", "plot_radial_acceptance"}:
        from .plot import plot_energy_slices, plot_radial_acceptance

        plotting_helpers = {
            "plot_energy_slices": plot_energy_slices,
            "plot_radial_acceptance": plot_radial_acceptance,
        }
        globals().update(plotting_helpers)
        return plotting_helpers[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
