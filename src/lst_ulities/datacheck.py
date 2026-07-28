from dataclasses import dataclass, replace
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class DataCheckTables:
    flatfield: pd.DataFrame
    cosmics_intensity_spectrum: pd.DataFrame
    runsummary: pd.DataFrame

    @classmethod
    def from_files(cls, files: List[str]) -> "DataCheckTables":
        """Load and concatenate DataCheck tables from a list of HDF5 files.

        Each file is opened once, and the tables ``flatfield``,
        ``cosmics_intensity_spectrum``, and ``runsummary`` are read and
        concatenated across files.

        Parameters
        ----------
        files : List[str]
            Paths to the HDF5 datacheck files.

        Returns
        -------
        DataCheckTables
            A dataclass instance holding the concatenated DataFrames.
        """
        table_names = list(cls.__annotations__.keys())
        table_data = {name: [] for name in table_names}

        for file in files:
            with pd.HDFStore(file, mode="r") as store:
                for name in table_names:
                    table_data[name].append(store[name])

        return cls(**{name: pd.concat(dataframes, ignore_index=True) for name, dataframes in table_data.items()})

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
                # Filter and reset the index
                filtered_fields[field_name] = val.loc[val["runnumber"].isin(runs_set)].reset_index(drop=True)  # pyright: ignore
            else:
                filtered_fields[field_name] = val

        return replace(self, **filtered_fields)
