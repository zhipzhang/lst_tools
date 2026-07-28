"""lst_ulities: utilities built on top of lstchain and ctapipe."""

from importlib.metadata import PackageNotFoundError, version

from .datacheck import DataCheckTables
from .datafilter import DataFilter
from .helper import glob_files
from .run_statistics import RunStatistics

__all__ = ["DataCheckTables", "DataFilter", "glob_files", "RunStatistics"]

try:
    __version__ = version("lst_ulities")
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"
