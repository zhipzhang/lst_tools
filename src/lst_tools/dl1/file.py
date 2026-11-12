"""A DL1 file and its path metadata."""

from functools import cached_property
from os import PathLike
from pathlib import Path

from lst_tools.finder import parse_dl1_path

from .nsb import resolve_dl1_nsb_level


class LSTDL1File:
    """One DL1 HDF5 file together with its path metadata.

    Construction parses the run number, tailcuts level, and LSTChain version
    from the path. The NSB level requires reading the run's DL1 data and is
    resolved lazily on first access.
    """

    def __init__(self, file_name: str | PathLike[str]):
        path = Path(file_name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        metadata = parse_dl1_path(str(path))
        self.file_name = metadata.path
        self.run_number = metadata.run_number
        self.tailcuts_level = metadata.tailcuts_level
        self.lstchain_version = metadata.lstchain_version

    @cached_property
    def nsb_level(self) -> float:
        """NSB tuning level estimated from the run's DL1 subrun data.

        The estimate is computed on first access and cached for later accesses.
        A failed estimate is not cached, so the next access retries it.
        """
        try:
            return float(resolve_dl1_nsb_level(self.file_name))
        except ValueError as error:
            raise RuntimeError(f"Failed to resolve NSB level: {error}") from error
