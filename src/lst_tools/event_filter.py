"""Utilities for selecting events."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import isnan

import numpy as np
import pandas as pd


def _validated_cut_bins(
    energy_low: Iterable[float],
    energy_high: Iterable[float],
    gh_cuts: Iterable[float],
) -> list[tuple[float, float, float]]:
    """Return validated ``(low, high, cut)`` tuples ordered by energy."""
    try:
        lows = tuple(energy_low)
        highs = tuple(energy_high)
        cuts = tuple(gh_cuts)
    except TypeError as error:
        raise ValueError("energy bounds and gammaness cuts must be iterable") from error

    if not (len(lows) == len(highs) == len(cuts)):
        raise ValueError("energy_low, energy_high, and gh_cuts must have the same length")

    bins: list[tuple[float, float, float]] = []
    for index, (low, high, cut) in enumerate(zip(lows, highs, cuts, strict=True)):
        try:
            cut_bin = (float(low), float(high), float(cut))
        except (TypeError, ValueError) as error:
            raise ValueError(f"cut bin {index} must contain numeric values") from error

        low_value, high_value, cut_value = cut_bin
        if any(isnan(value) for value in cut_bin):
            raise ValueError(f"cut bin {index} must not contain NaN")
        if low_value >= high_value:
            raise ValueError(f"energy_low must be less than energy_high in cut bin {index}")
        bins.append((low_value, high_value, cut_value))

    bins.sort(key=lambda cut_bin: cut_bin[0])
    for previous, current in pairwise(bins):
        if current[0] < previous[1]:
            raise ValueError("energy bins must not overlap")

    return bins


def get_energy_dependent_gammaness_cuts(
    data: pd.DataFrame,
    energy_edges: Iterable[float],
    survival_fraction: float,
) -> np.ndarray:
    """Compute gammaness cuts for a target survival fraction in each energy bin.

    Parameters
    ----------
    data
        Event data containing ``reco_energy`` and ``gammaness`` columns.
    energy_edges
        Energy-bin edges.
    survival_fraction
        Fraction of events to retain above the gammaness threshold.
        Must be between 0 and 1.

    Returns
    -------
    np.ndarray
        Gammaness cut for each energy bin. Bins without events get NaN.
    """
    if not 0 < survival_fraction <= 1:
        raise ValueError("survival_fraction must be in (0, 1]")

    energy_edges = np.asarray(energy_edges, dtype=float)
    cuts = np.full(len(energy_edges) - 1, np.nan)

    energy = data["reco_energy"]
    gammaness = data["gammaness"]

    for i, (low, high) in enumerate(pairwise(energy_edges)):
        mask = energy.ge(low) & energy.lt(high) & gammaness.notna()

        values = gammaness.loc[mask].to_numpy()

        if len(values) == 0:
            continue

        cuts[i] = np.quantile(
            values,
            1 - survival_fraction,
        )

    return cuts


def apply_energy_dependent_gammaness_cuts(
    data: pd.DataFrame,
    energy_low: Iterable[float],
    energy_high: Iterable[float],
    gh_cuts: Iterable[float],
) -> pd.DataFrame:
    """Return events that pass an energy-dependent gammaness cut.

    Each entry in ``energy_low``, ``energy_high``, and ``gh_cuts`` describes
    one energy interval. Intervals include their lower edge and exclude their
    upper edge. Within an interval, an event passes when its ``gammaness`` is
    greater than or equal to the corresponding cut.

    Events outside the configured intervals, or with missing energy or
    gammaness values, are excluded. The returned frame retains the input row
    order, index, and columns.

    Parameters
    ----------
    data
        Event data containing ``reco_energy`` and ``gammaness`` columns.
    energy_low, energy_high
        Lower and upper energy bounds for each interval.
    gh_cuts
        Minimum gammaness for each interval.

    Raises
    ------
    ValueError
        If required columns are absent or the cut configuration is invalid.
    """
    required_columns = {"reco_energy", "gammaness"}
    if not required_columns.issubset(data.columns):
        raise ValueError("DataFrame must contain 'reco_energy' column and 'gammaness' column")

    bins = _validated_cut_bins(energy_low, energy_high, gh_cuts)
    selected = pd.Series(False, index=data.index, dtype=bool)

    for low, high, cut in bins:
        in_energy_bin = data["reco_energy"].ge(low) & data["reco_energy"].lt(high)
        selected |= in_energy_bin & data["gammaness"].ge(cut)

    return data.loc[selected]


@dataclass
class EventFilter:
    intensity_cuts: float
    energy_low: np.ndarray
    energy_high: np.ndarray
    gh_cuts: np.ndarray

    @classmethod
    def from_e_edges(cls, intensity_cuts: float, energy_edges: np.ndarray, gh_cuts: np.ndarray):
        return cls(intensity_cuts, energy_edges[:-1], energy_edges[1:], gh_cuts)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        after_gh_cuts = apply_energy_dependent_gammaness_cuts(data, self.energy_low, self.energy_high, self.gh_cuts)
        after_intensity_cuts = after_gh_cuts.loc[after_gh_cuts["intensity"].ge(self.intensity_cuts)]
        return after_intensity_cuts
