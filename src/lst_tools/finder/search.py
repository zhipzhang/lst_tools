"""
Search for already-produced LST data files on the La Palma server.
"""

import glob
import warnings

from .search_results import (
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3LevelSearchResult,
)

LAPLAMA_DATA_PATHS = ("/fefs/onsite/data/lst-pipe/LSTN-01", "/fefs/aswg/data/real")

VALID_LEVELS = ("DL1", "DL2", "DL3")


def _glob_run_files(date: int, run_number: int, level: str) -> list[str]:
    ext = "fits" if level == "DL3" else "h5"
    file_name = f"{level.lower()}_LST-1.Run{run_number:05d}.{ext}"

    matched_files = []
    for base_path in LAPLAMA_DATA_PATHS:
        matched_files.extend(glob.glob(f"{base_path}/{level}/{date}/**/{file_name}", recursive=True))
    return sorted(matched_files)


def find_laplama_files(
    date: int, run_number: int, level: str = "dl1"
) -> DL1LevelSearchResult | DL2LevelSearchResult | DL3LevelSearchResult:
    """Search the La Palma server for existing files of one run.

    Returns the search result for the requested data level. Use the
    ``newest_attributes`` property on it: DL1/DL2 give the single file with
    the newest lstchain version (``None`` if nothing was found), DL3 gives
    the list of files produced with the newest version.
    """
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level: {level}. Must be 'DL1', 'DL2', or 'DL3'.")

    matched_files = _glob_run_files(date, run_number, level)

    if len(matched_files) == 0:
        warnings.warn(
            f"No file found for {level}_LST-1.Run{run_number:05d} under any of {LAPLAMA_DATA_PATHS}",
            stacklevel=2,
        )

    if level == "DL1":
        return DL1LevelSearchResult(matched_files)
    if level == "DL2":
        return DL2LevelSearchResult(matched_files)
    return DL3LevelSearchResult(matched_files)
