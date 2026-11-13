"""Filtering DL2 event tables with lstchain quality cuts and gammaness cuts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table
from lstchain.io.event_selection import EventSelector


def _gammaness_column(data: pd.DataFrame) -> str:
    if "gh_score" in data.columns:
        return "gh_score"
    if "gammaness" in data.columns:
        return "gammaness"
    raise ValueError("DataFrame must contain 'gh_score' or 'gammaness'")


def _energy_column_to_tev(column) -> np.ndarray:
    if hasattr(column, "unit") and column.unit is not None:
        return np.asarray(u.Quantity(column).to_value(u.TeV), dtype=float)
    return np.asarray(column, dtype=float)


@dataclass(frozen=True)
class EnergyDependentGHCut:
    """Energy-binned gammaness cut thresholds (energies in TeV)."""

    energy_low: np.ndarray
    energy_high: np.ndarray
    cuts: np.ndarray

    def __post_init__(self) -> None:
        if len(self.energy_low) == 0:
            raise ValueError("at least one energy bin is required")
        if not (len(self.energy_low) == len(self.energy_high) == len(self.cuts)):
            raise ValueError("energy_low, energy_high, and cuts must have the same length")

    @classmethod
    def from_edges(cls, energy_edges: Sequence[float] | np.ndarray, cuts: Sequence[float] | np.ndarray) -> Self:
        """Create binned gammaness cuts from contiguous energy bin edges in TeV."""
        edges = np.asarray(energy_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("energy_edges must be a strictly increasing 1D sequence with at least two entries")
        return cls(energy_low=edges[:-1], energy_high=edges[1:], cuts=np.asarray(cuts, dtype=float))

    @classmethod
    def from_table(cls, cut_table: Table) -> Self:
        """Create binned gammaness cuts from a lstchain ``GH_CUTS`` table."""
        missing = {"low", "high", "cut"}.difference(cut_table.colnames)
        if missing:
            raise ValueError(f"cut_table is missing required columns: {sorted(missing)}")
        return cls(
            energy_low=_energy_column_to_tev(cut_table["low"]),
            energy_high=_energy_column_to_tev(cut_table["high"]),
            cuts=np.asarray(cut_table["cut"], dtype=float),
        )

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return events falling inside an energy bin with gammaness ``>= cut``."""
        if "reco_energy" not in data.columns:
            raise ValueError("DataFrame must contain 'reco_energy'")
        gh_col = _gammaness_column(data)

        energy = data["reco_energy"]
        gh_score = data[gh_col]
        mask = pd.Series(False, index=data.index, dtype=bool)
        for low, high, cut in zip(self.energy_low, self.energy_high, self.cuts, strict=True):
            mask |= energy.ge(low) & energy.lt(high) & gh_score.ge(cut)
        return data.loc[mask]


class DL2EventFilter:
    """Filter a DL2 event table with lstchain ``EventSelector`` and gammaness cuts.

    Applies ``EventSelector.filter_cut`` first and optionally a gammaness cut:
    ``None`` (quality cuts only), a fixed scalar threshold (``gh > cut``), or an
    :class:`EnergyDependentGHCut` (``gh >= bin_cut``).
    """

    def __init__(
        self,
        event_selector: EventSelector | None = None,
        gh_cut: float | EnergyDependentGHCut | None = None,
    ) -> None:
        self.event_selector = event_selector if event_selector is not None else EventSelector()
        if isinstance(gh_cut, bool) or (
            gh_cut is not None and not isinstance(gh_cut, (int, float, EnergyDependentGHCut))
        ):
            raise TypeError("gh_cut must be a float, EnergyDependentGHCut, or None")
        self.gh_cut: float | EnergyDependentGHCut | None = (
            float(gh_cut) if isinstance(gh_cut, (int, float)) else gh_cut
        )

    @classmethod
    def from_cuts(
        cls,
        intensity_cut: float | None = None,
        leakage_cut: float | None = None,
        *,
        gh_cut: float | EnergyDependentGHCut | None = None,
    ) -> Self:
        """Build a filter from scalar quality cuts and an optional gammaness cut."""
        filters = dict(EventSelector().filters)
        if intensity_cut is not None:
            filters["intensity"] = [float(intensity_cut), np.inf]
        if leakage_cut is not None:
            filters["leakage_intensity_width_2"] = [0, float(leakage_cut)]
        return cls(EventSelector(filters=filters), gh_cut=gh_cut)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return events passing quality filters and the configured gammaness cut."""
        selected = self.event_selector.filter_cut(data.copy())
        if self.gh_cut is None:
            return selected
        if isinstance(self.gh_cut, EnergyDependentGHCut):
            return self.gh_cut.apply(selected)
        gh_col = _gammaness_column(selected)
        return selected.loc[selected[gh_col] > self.gh_cut]
