"""Tools for camera-coordinate background studies."""

from .camera import CameraImage
from .events import SkyOffsetEvents
from .lst_adapter import sky_offset_events_from_lstdl2

__all__ = [
    "Background2DMaker",
    "CameraImage",
    "SkyOffsetEvents",
    "plot_energy_slices",
    "plot_radial_acceptance",
    "sky_offset_events_from_lstdl2",
]


def __getattr__(name: str):
    """Load Matplotlib-based helpers only when requested."""
    if name == "Background2DMaker":
        from .background2d import Background2DMaker

        globals()[name] = Background2DMaker
        return Background2DMaker
    if name in {"plot_energy_slices", "plot_radial_acceptance"}:
        from .plot import plot_energy_slices, plot_radial_acceptance

        plotting_helpers = {
            "plot_energy_slices": plot_energy_slices,
            "plot_radial_acceptance": plot_radial_acceptance,
        }
        globals().update(plotting_helpers)
        return plotting_helpers[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
