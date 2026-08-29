"""Lazy, process-wide access to an analysis's DataCheck products."""

from pathlib import Path

from .datacheck import DataCheckTables
from .run_statistics import RunStatistics


class DataCheckStore:
    """Lazy access to DataCheck products stored in one analysis directory.

    Initialization records only the directory. Each product is loaded from its
    own HDF5 file the first time that product is accessed, then cached for the
    rest of the process.
    """

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._data_check_pattern = "data_check_*.h5"
        self._run_statistics_pattern = "selected_runs_*.h5"
        self._data_check_tables: DataCheckTables | None = None
        self._run_statistics: RunStatistics | None = None

    def initialize(
        self,
        directory: str | Path,
        *,
        data_check_pattern: str = "data_check_*.h5",
        run_statistics_pattern: str = "selected_runs_*.h5",
    ) -> "DataCheckStore":
        """Point the store at a DataCheck output directory without reading it."""
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(f"DataCheck directory does not exist: {directory}")

        self._directory = directory
        self._data_check_pattern = data_check_pattern
        self._run_statistics_pattern = run_statistics_pattern
        self._data_check_tables = None
        self._run_statistics = None
        return self

    @property
    def is_initialized(self) -> bool:
        """Whether a DataCheck output directory has been configured."""
        return self._directory is not None

    @property
    def directory(self) -> Path:
        """Configured DataCheck output directory."""
        if self._directory is None:
            raise RuntimeError(
                "Global DataCheck store is not initialized. "
                "Call initialize_data_check(directory) first."
            )
        return self._directory

    def _matching_file(self, pattern: str, product_name: str) -> Path:
        files = sorted(self.directory.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No {product_name} file matching {pattern!r} found in {self.directory}"
            )
        if len(files) > 1:
            raise RuntimeError(
                f"Multiple {product_name} files matched {pattern!r} in "
                f"{self.directory}: {files}"
            )
        return files[0]

    @property
    def data_check_tables(self) -> DataCheckTables:
        """Lazily load and cache ``data_check_*.h5`` tables."""
        if self._data_check_tables is None:
            file = self._matching_file(self._data_check_pattern, "DataCheck tables")
            self._data_check_tables = DataCheckTables.from_files([str(file)])
        return self._data_check_tables

    @property
    def run_statistics(self) -> RunStatistics:
        """Lazily load and cache the separate ``selected_runs_*.h5`` file."""
        if self._run_statistics is None:
            file = self._matching_file(self._run_statistics_pattern, "RunStatistics")
            self._run_statistics = RunStatistics.from_file(str(file))
        return self._run_statistics


run_data_check = DataCheckStore()
"""The process-wide, lazily loaded DataCheck store."""


def initialize_data_check(
    directory: str | Path,
    *,
    data_check_pattern: str = "data_check_*.h5",
    run_statistics_pattern: str = "selected_runs_*.h5",
) -> DataCheckStore:
    """Initialize and return the process-wide lazy DataCheck store."""
    return run_data_check.initialize(
        directory,
        data_check_pattern=data_check_pattern,
        run_statistics_pattern=run_statistics_pattern,
    )
