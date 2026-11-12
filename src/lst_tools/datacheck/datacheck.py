"""Load LST-1 DL1 DataCheck HDF5 files and derive per-run statistics.

This module targets the LST-1 DL1 DataCheck layout with the ``flatfield``,
``cosmics_intensity_spectrum``, and ``runsummary`` tables. Other instruments
or DataCheck formats require a different loader and statistics specification.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import circular_mean_degrees, sliding_window_mode

logger = logging.getLogger(__name__)

# Public per-run statistics schema consumed by RunQualityFilter. Keys are derived
# column names; values identify the source column and aggregation for each input table.
DEFAULT_SPEC = {
    "cosmics_intensity_spectrum": {
        "n_subruns": ("runnumber", "size"),
        "date": ("yyyymmdd", "first"),
        "mean_R422": ("ZD_corrected_cosmics_rate_at_422_pe", "mean"),
        "std_R422": ("ZD_corrected_cosmics_rate_at_422_pe", "std"),
        "mode_R422": ("ZD_corrected_cosmics_rate_at_422_pe", lambda x: sliding_window_mode(x)),
        "fraction_around_mode_R422": (
            "ZD_corrected_cosmics_rate_at_422_pe",
            lambda x: sliding_window_mode(x, return_fraction=True),
        ),
        "mean_intensity_at_reference_rate": ("intensity_at_reference_rate", "mean"),
        "std_intensity_at_reference_rate": ("intensity_at_reference_rate", "std"),
        "mean_light_yield": ("light_yield", "mean"),
        "std_light_yield": ("light_yield", "std"),
        "mean_index": ("ZD_corrected_cosmics_spectral_index", "mean"),
        "std_index": ("ZD_corrected_cosmics_spectral_index", "std"),
        "mean_fit_p_value": ("intensity_spectrum_fit_p_value", "mean"),
        "mean_intensity_threshold": ("ZD_corrected_intensity_at_half_peak_rate", "mean"),
        "std_intensity_threshold": ("ZD_corrected_intensity_at_half_peak_rate", "std"),
        "mean_ra": ("ra_tel", lambda x: circular_mean_degrees(x)),
        "mean_dec": ("dec_tel", "mean"),
        "pointing_dec_std": ("dec_tel", "std"),
        "mean_cos_zd": ("cos_zenith", "mean"),
        "mean_diffuse_nsb_std": ("diffuse_nsb_std", "mean"),
    },
    "runsummary": {
        "n_flatfield": ("num_flatfield", "first"),
        "n_pedestal": ("num_pedestals", "first"),
        "observation_time": ("elapsed_time", "first"),
    },
}


@dataclass
class DataCheckTables:
    flatfield: pd.DataFrame
    cosmics_intensity_spectrum: pd.DataFrame
    runsummary: pd.DataFrame

    @classmethod
    def from_files(cls, files: list[str]) -> "DataCheckTables":
        """Load and concatenate DataCheck tables from a list of HDF5 files.

        Each file is opened once, and the tables ``flatfield``,
        ``cosmics_intensity_spectrum``, and ``runsummary`` are read and
        concatenated across files. Files missing any required table are
        skipped with a warning so that incomplete nightly products do not
        prevent run discovery from the remaining files.

        Parameters
        ----------
        files : List[str]
            Paths to the HDF5 datacheck files.

        Returns
        -------
        DataCheckTables
            A dataclass instance holding the concatenated DataFrames.
        """
        # The dataclass fields intentionally define the LST-1 table schema used by
        # loading, saving, and run selection. Revisit if another DataCheck layout is supported.
        table_names = list(cls.__annotations__.keys())
        table_data: dict[str, list[pd.DataFrame]] = {name: [] for name in table_names}

        for file in files:
            with pd.HDFStore(file, mode="r") as store:
                available = set(store.keys())
                missing = [name for name in table_names if f"/{name}" not in available]
                if missing:
                    logger.warning("Skipping %s: missing table(s) %s", file, ", ".join(f"/{name}" for name in missing))
                    continue

                for name in table_names:
                    table_data[name].append(store[f"/{name}"])

        if not table_data[table_names[0]]:
            raise ValueError("No valid DataCheck files were loaded")

        return cls(**{name: pd.concat(dataframes, ignore_index=True) for name, dataframes in table_data.items()})

    def get_statistics(self, spec: dict) -> pd.DataFrame:
        """Aggregate tables by run, preserving runs missing from any one table."""
        table_statistics = []

        for table_name, table_spec in spec.items():
            if table_name not in self.__dataclass_fields__:
                raise ValueError(f"Unknown DataCheck table in statistics spec: {table_name!r}")
            table = getattr(self, table_name)
            required_columns = {column for column, _ in table_spec.values()} | {"runnumber"}
            missing_columns = required_columns.difference(table.columns)
            if missing_columns:
                raise ValueError(
                    f"Statistics spec for {table_name!r} references missing columns: {sorted(missing_columns)}"
                )
            table_statistics.append(table.groupby("runnumber").agg(**table_spec))

        return pd.concat(table_statistics, axis="columns", join="outer").sort_index().reset_index()

    # Input tables are treated as immutable after construction, so one cached result is enough.
    @cached_property
    def statistics(self) -> pd.DataFrame:
        """Return and cache the run-wise statistics defined by :data:`DEFAULT_SPEC`."""
        return self.get_statistics(DEFAULT_SPEC)

    def save_to_h5file(self, file: str | Path, overwrite: bool = False) -> None:
        """Save all DataCheck tables to an HDF5 file.

        By default existing tables are preserved and new rows are appended.
        Set ``overwrite=True`` to replace the complete file.
        """
        mode = "w" if overwrite else "a"
        with pd.HDFStore(file, mode=mode) as store:
            for name in self.__dataclass_fields__:
                table = getattr(self, name)
                if overwrite:
                    store.put(name, table, format="fixed" if table.empty else "table")
                elif table.empty:
                    if f"/{name}" not in store:
                        store.put(name, table, format="fixed")
                elif f"/{name}" in store and not store.get_storer(name).is_table:
                    existing = store[name]
                    store.put(name, pd.concat([existing, table], ignore_index=True), format="table")
                else:
                    store.append(name, table)

    def describe(self) -> None:
        """Print a summary of the loaded runs.

        The summary shows the total number of runs, the run-number range, and
        the calendar-date range covered by the datacheck data.
        """
        run_numbers, first_indices = np.unique(self.cosmics_intensity_spectrum["runnumber"], return_index=True)
        dates = pd.to_datetime(
            self.cosmics_intensity_spectrum["yyyymmdd"].iloc[first_indices],
            format="%Y%m%d",
        )

        n_runs = len(run_numbers)
        run_min, run_max = int(run_numbers.min()), int(run_numbers.max())
        date_min = dates.min().strftime("%Y-%m-%d")
        date_max = dates.max().strftime("%Y-%m-%d")

        labels = ["Number of runs", "Run number range", "Date range"]
        values = [
            str(n_runs),
            f"{run_min} → {run_max}",
            f"{date_min} → {date_max}",
        ]
        label_width = max(len(label) for label in labels)
        value_width = max(len(value) for value in values)

        h = "─"
        top = "┌─" + h * label_width + "─┬─" + h * value_width + "─┐"
        sep = "├─" + h * label_width + "─┼─" + h * value_width + "─┤"
        bot = "└─" + h * label_width + "─┴─" + h * value_width + "─┘"

        print(top)
        for i, (label, value) in enumerate(zip(labels, values)):
            if i:
                print(sep)
            print(f"│ {label:<{label_width}} │ {value:>{value_width}} │")
        print(bot)

    def select_runs(self, runnumbers: Iterable[int]) -> "DataCheckTables":
        """Filter all tables by run-number membership, returning a new DataCheckTables (original unchanged)."""
        # Normalize to a set for fast hash-based lookups; also accepts generators, lists, etc.
        runs_set = set(runnumbers) if not isinstance(runnumbers, set) else runnumbers

        # Only process DataFrame fields explicitly for robustness
        filtered_fields = {}
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, pd.DataFrame):
                # Keep runnumber as a data column and discard only the old positional index,
                # giving each filtered table a fresh RangeIndex without an "index" column.
                filtered_fields[field_name] = val.loc[val["runnumber"].isin(runs_set)].reset_index(drop=True)  # pyright: ignore
            else:
                filtered_fields[field_name] = val

        return replace(self, **filtered_fields)
