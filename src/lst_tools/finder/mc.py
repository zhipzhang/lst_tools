"""Search for Monte Carlo DL2 files in the standard AllSky layout."""

from pathlib import Path

from .models import MCDL2Metadata
from .parsing.mc import parse_mc_dl2_path

MC_DL2_PATH = Path("/fefs/aswg/data/mc/DL2/AllSky")


def find_mc_dl2_files(base_directory: str | Path) -> list[MCDL2Metadata]:
    """Recursively find and parse merged DL2 files below one MC campaign."""
    base_directory = Path(base_directory)
    if not base_directory.exists():
        raise FileNotFoundError(base_directory)
    if not base_directory.is_dir():
        raise NotADirectoryError(base_directory)

    paths = sorted(path for path in base_directory.rglob("dl2_*merged.h5") if path.is_file())
    return [parse_mc_dl2_path(path) for path in paths]
