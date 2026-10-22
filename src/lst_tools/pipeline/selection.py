"""
Run selection: the interface every pipeline stage consumes.

``RunSelection`` is produced by run selection (datacheck tables + data
quality cuts + zenith binning) and consumed by every producer. Consumers
iterate it for typed :class:`SelectedRun` records — the underlying
DataFrame is storage, not a public API.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lst_tools.datacheck import DataCheckTables, DataFilter


@dataclass(frozen=True)
class SelectedRun:
    """One run that passed selection, with everything later stages need."""

    run_number: int
    date: int
    zenith_bin: str


@dataclass
class RunSelection:
    """The selected runs and the bookkeeping behind them.

    ``selected`` is the run table with the ``zenith_bin`` column (``None``
    for runs that passed the cuts but got no bin). Consumers should
    iterate the selection rather than read the frame directly.
    """

    edges: tuple[float, ...]
    selected: pd.DataFrame
    tables: "DataCheckTables | None" = None
    after_basic_cuts: pd.DataFrame | None = None
    data_filter: "DataFilter | None" = None

    def __iter__(self) -> Iterator[SelectedRun]:
        """The selected runs as typed records, binned runs only."""
        binned = self.selected[self.selected["zenith_bin"].map(lambda value: isinstance(value, str))]
        for record in binned.to_dict(orient="records"):
            yield SelectedRun(
                run_number=int(record["run_number"]),
                date=int(record["date"]),
                zenith_bin=record["zenith_bin"],
            )

    @property
    def unbinned_run_numbers(self) -> list[int]:
        """Runs that passed the cuts but got no zenith bin."""
        unbinned = self.selected[self.selected["zenith_bin"].map(lambda value: not isinstance(value, str))]
        return [int(run_number) for run_number in unbinned["run_number"].tolist()]

    def __repr__(self) -> str:
        return (
            f"RunSelection({len(list(self))} binned runs, "
            f"{len(self.unbinned_run_numbers)} unbinned, edges={self.edges})"
        )
