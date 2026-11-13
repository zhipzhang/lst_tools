"""Run selection: the interface every pipeline stage consumes.

``RunSelection`` combines the selected run table with the bookkeeping needed
for datacheck outputs. Consumers iterate it for typed :class:`SelectedRun`
records rather than depending on the DataFrame storage format.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lst_tools.datacheck import DataCheckTables, RunQualityFilter


@dataclass(frozen=True)
class SelectedRun:
    """One run that passed selection, with everything later stages need."""

    run_number: int
    date: int


@dataclass
class RunSelection:
    """The selected runs and the bookkeeping behind them."""

    selected: pd.DataFrame
    tables: "DataCheckTables | None" = None
    after_basic_cuts: pd.DataFrame | None = None
    data_filter: "RunQualityFilter | None" = None

    def __iter__(self) -> Iterator[SelectedRun]:
        """Yield every selected run as a typed record."""
        for record in self.selected.to_dict(orient="records"):
            yield SelectedRun(
                run_number=int(record["runnumber"]),
                date=int(record["date"]),
            )

    def __repr__(self) -> str:
        return f"RunSelection({len(self.selected)} runs)"
