"""Utilities for selecting events."""

import numpy as np
import pandas as pd
from lstchain.io.event_selection import DataBinning, DL3Cuts, EventSelector

# DL3Cuts values that differ from lstchain defaults, matching
# irf/default_config.json.
_TUNED_DL3_CUTS = {
    "gh_efficiency": 0.7,
    "max_gh_cut": 0.98,
    "global_alpha_cut": 10,
    "theta_containment": 0.7,
    "alpha_containment": 0.7,
}


class EventFilter:
    """Event selection for DL2 filtering and IRF generation.

    Wraps lstchain's ``EventSelector`` (quality filters) and ``DL3Cuts``
    (gammaness cut); the same component instances are used to filter DL2
    events and to write the IRF-generation config, so both always agree.
    Like lstchain's own tools, the gammaness mode is chosen by the
    ``energy_dependent_gh`` flag: a fixed ``global_gh_cut`` when off, or
    per-energy-bin quantile cuts at ``gh_efficiency`` when on.

    Effective defaults (lstchain's, unless created via :meth:`from_cuts`):

    - ``EventSelector.filters``: ``intensity`` ``[<intensity_cut>, inf]``,
      ``leakage_intensity_width_2`` ``[0, <leakage_cut>]``, ``r`` ``[0, 1]``,
      ``wl`` ``[0.01, 1]``, ``event_type`` ``[32, 32]``
    - ``EventSelector.finite_params``: ``[intensity, length, width]``
    - ``DL3Cuts``: ``min_gh_cut 0.1``, ``max_gh_cut 0.98``,
      ``min_event_p_en_bin 100`` (used to derive the quantile cuts)

    Filters frames carrying a ``gh_score`` column (present on all
    ``LSTDL2`` tables) and, in energy-dependent mode, ``reco_energy`` in
    TeV. Note that lstchain's fill logic requires at least one energy bin
    with ``min_event_p_en_bin`` events in energy-dependent mode.
    """

    def __init__(
        self,
        event_selector: EventSelector,
        dl3_cuts: DL3Cuts,
        *,
        energy_dependent_gh: bool = False,
        energy_bins: np.ndarray | None = None,
    ):
        self.event_selector = event_selector
        self.dl3_cuts = dl3_cuts
        self.energy_dependent_gh = energy_dependent_gh
        if energy_bins is None:
            energy_bins = DataBinning().reco_energy_bins().to_value("TeV")
        self.energy_bins = np.asarray(energy_bins, dtype=float)

    @classmethod
    def from_cuts(
        cls,
        intensity_cut: float,
        leakage_cut: float,
        *,
        gh_cut: float | None = None,
        gh_efficiency: float | None = None,
        energy_bins: np.ndarray | None = None,
    ) -> "EventFilter":
        """Create a filter from scalar cut values.

        Exactly one gammaness mode must be given: ``gh_cut`` for a fixed
        threshold, or ``gh_efficiency`` for per-energy-bin quantile cuts
        derived from the filtered data on ``energy_bins`` (defaults to the
        ``DataBinning`` reco-energy binning, in TeV).
        """
        if (gh_cut is None) == (gh_efficiency is None):
            raise ValueError("exactly one of gh_cut and gh_efficiency must be set")
        if gh_efficiency is not None and not 0 < gh_efficiency <= 1:
            raise ValueError("gh_efficiency must be in (0, 1]")

        event_selector = EventSelector(
            filters={
                **EventSelector().filters,
                "intensity": [float(intensity_cut), np.inf],
                "leakage_intensity_width_2": [0, float(leakage_cut)],
                "width": [0.0, np.inf],
                "length": [0.0, np.inf],
            }
        )
        if gh_cut is not None:
            dl3_cuts = DL3Cuts(**{**_TUNED_DL3_CUTS, "global_gh_cut": float(gh_cut)})
        else:
            dl3_cuts = DL3Cuts(**{**_TUNED_DL3_CUTS, "gh_efficiency": float(gh_efficiency)})

        return cls(
            event_selector,
            dl3_cuts,
            energy_dependent_gh=gh_efficiency is not None,
            energy_bins=energy_bins,
        )

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return events passing the configured cuts."""
        selected = self.event_selector.filter_cut(data).copy()
        if not self.energy_dependent_gh:
            return self.dl3_cuts.apply_global_gh_cut(selected)

        gh_cuts = self.dl3_cuts.energy_dependent_gh_cuts(selected, self.energy_bins)
        return self.dl3_cuts.apply_energy_dependent_gh_cuts(selected, gh_cuts)

    def to_lstchain_config(self) -> dict:
        """Return the lstchain config sections defined by this filter."""
        return {
            "EventSelector": {name: getattr(self.event_selector, name) for name in self.event_selector.traits(config=True)},
            "DL3Cuts": {name: getattr(self.dl3_cuts, name) for name in self.dl3_cuts.traits(config=True)},
        }
