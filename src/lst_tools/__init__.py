"""lst_tools: utilities built on top of lstchain and ctapipe."""

from importlib.metadata import PackageNotFoundError, version

from .datacheck import (
    DataCheckStore,
    DataCheckTables,
    DataFilter,
    RunStatistics,
    initialize_data_check,
    plot_advanced_distributions,
    run_data_check,
)
from .helper import glob_files, plot_histogram

__all__ = [
    "DataCheckStore",
    "DataCheckTables",
    "DataFilter",
    "RunStatistics",
    "glob_files",
    "initialize_data_check",
    "plot_advanced_distributions",
    "plot_histogram",
    "run_data_check",
]

try:
    __version__ = version("lst_tools")
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"
