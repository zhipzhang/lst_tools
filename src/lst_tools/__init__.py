"""lst_tools: utilities built on top of lstchain and ctapipe."""

from importlib.metadata import PackageNotFoundError, version

from .datacheck import (
    DataCheckTables,
    DataFilter,
    assign_zenith_bins,
)
from .helper import glob_files, plot_histogram

__all__ = [
    "DataCheckTables",
    "DataFilter",
    "assign_zenith_bins",
    "glob_files",
    "plot_advanced_distributions",
    "plot_histogram",
]

try:
    __version__ = version("lst_tools")
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"


def __getattr__(name: str):
    """Load plotting support only when it is requested."""
    if name == "plot_advanced_distributions":
        from .datacheck import plot_advanced_distributions

        globals()[name] = plot_advanced_distributions
        return plot_advanced_distributions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
