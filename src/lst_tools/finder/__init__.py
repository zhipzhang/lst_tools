"""
In LST Data Analysis, we will always need to search the data files for different data levels.
Here we wrapper those functions in here, to provide a convenient interface for searching files.

The supported search includes:
    - use date range to filter the datacheck file, so that we don't need to load all files
    - use `find_la_palma_files` to search files already produced on the La Palma server
    - use `DataLevelSearchResult` to return the search results for different data levels
"""

from .datalevel_attributes import DLAttributes, NSBLevel
from .helper import parse_dl1_path, parse_dl2_path, parse_dl3_path
from .search import LA_PALMA_DATA_PATHS, find_la_palma_files
from .search_results import (
    DataLevelSearchResult,
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3LevelSearchResult,
)

__all__ = [
    "LA_PALMA_DATA_PATHS",
    "DL1LevelSearchResult",
    "DL2LevelSearchResult",
    "DL3LevelSearchResult",
    "DLAttributes",
    "DataLevelSearchResult",
    "NSBLevel",
    "find_la_palma_files",
    "parse_dl1_path",
    "parse_dl2_path",
    "parse_dl3_path",
]
