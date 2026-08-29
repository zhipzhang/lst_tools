"""lst_ulities: utilities built on top of lstchain and ctapipe."""

from importlib.metadata import PackageNotFoundError, version

from .datacheck import (
    DataCheckStore,
    DataCheckTables,
    DataFilter,
    RunStatistics,
    run_data_check,
    initialize_data_check,
)
from .helper import glob_files, plot_histogram

__all__ = [
    "run_data_check",
    "DataCheckStore",
    "DataCheckTables",
    "DataFilter",
    "glob_files",
    "initialize_data_check",
    "plot_histogram",
    "RunStatistics",
]

try:
    __version__ = version("lst_ulities")
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"
