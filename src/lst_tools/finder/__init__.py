"""Find LST data files and parse their metadata.

This package provides the stable public interface for observed and Monte Carlo
data discovery, path parsing, and search results.
"""

from lst_tools.nsb import NSBLevel, find_closest_nsb_level

from .mc import MC_DL2_PATH, find_mc_dl2_files
from .models import DL1Metadata, DL2Metadata, DL3Metadata, MCDL2Metadata, ObservedDataMetadata
from .observed import LA_PALMA_DATA_PATHS, find_la_palma_files, find_la_palma_run
from .parsing.mc import parse_mc_dl2_path
from .parsing.observed import parse_dl1_path, parse_dl2_path, parse_dl3_path, parse_run_id
from .results import (
    DataLevelSearchResult,
    DataSearchResult,
    DL1LevelSearchResult,
    DL1SearchResult,
    DL2LevelSearchResult,
    DL2SearchResult,
    DL3LevelSearchResult,
    DL3SearchResult,
)


def filter_date(files: list[str], start_date: int, end_date: int) -> list[str]:
    from lst_tools.datacheck.utils import filter_paths_by_date

    return filter_paths_by_date(files, start_date, end_date)


__all__ = [
    "LA_PALMA_DATA_PATHS",
    "MC_DL2_PATH",
    "DL1LevelSearchResult",
    "DL1Metadata",
    "DL1SearchResult",
    "DL2LevelSearchResult",
    "DL2Metadata",
    "DL2SearchResult",
    "DL3LevelSearchResult",
    "DL3Metadata",
    "DL3SearchResult",
    "DataLevelSearchResult",
    "DataSearchResult",
    "MCDL2Metadata",
    "NSBLevel",
    "ObservedDataMetadata",
    "filter_date",
    "find_closest_nsb_level",
    "find_la_palma_files",
    "find_la_palma_run",
    "find_mc_dl2_files",
    "parse_dl1_path",
    "parse_dl2_path",
    "parse_dl3_path",
    "parse_mc_dl2_path",
    "parse_run_id",
]
