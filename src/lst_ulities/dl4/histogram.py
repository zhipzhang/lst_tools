"""Create reconstructed-energy count histograms for on/off regions."""

from collections.abc import Sequence

import astropy.units as u
import hist
import numpy as np
from gammapy.data import Observation, Observations
from gammapy.maps import MapAxis
from regions import CircleSkyRegion


class HistogramCountsOnoff:
    """Accumulate on/off event counts as a function of estimated energy.

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Signal region.
    off_region : sequence of `~regions.CircleSkyRegion`
        Background-control regions. Events falling in any of these regions are
        entered once in the off histogram.
    energy_axis : `~gammapy.maps.MapAxis` or `~astropy.units.Quantity`
        Energy-bin edges. Unitless input is interpreted as TeV.

    Notes
    -----
    Calling an instance adds the supplied observations to the existing
    histograms. Use :meth:`reset` before a call when a fresh result is needed.
    """

    def __init__(
        self,
        on_region: CircleSkyRegion,
        off_region: Sequence[CircleSkyRegion],
        energy_axis: MapAxis | u.Quantity,
    ):
        self.on_region = on_region
        self.off_region = tuple(off_region)

        if hasattr(energy_axis, "edges"):
            self.energy_bins = energy_axis.edges.to_value("TeV")
        elif isinstance(energy_axis, u.Quantity):
            self.energy_bins = energy_axis.to_value("TeV")
        else:
            self.energy_bins = np.asarray(energy_axis, dtype=float)

        self.energy_bins = np.asarray(self.energy_bins, dtype=float)
        if self.energy_bins.ndim != 1 or len(self.energy_bins) < 2:
            raise ValueError("energy_axis must contain at least two one-dimensional bin edges")
        if not np.all(np.isfinite(self.energy_bins)):
            raise ValueError("energy_axis bin edges must be finite")
        if np.any(np.diff(self.energy_bins) <= 0):
            raise ValueError("energy_axis bin edges must be strictly increasing")

        self.n_off = len(self.off_region)
        self.alpha = 1.0 / self.n_off if self.n_off > 0 else 0.0

        self.h_on = self._make_histogram()
        self.h_off = self._make_histogram()

    def _make_histogram(self) -> hist.Hist:
        energy_axis = hist.axis.Variable(
            self.energy_bins,
            name="estimated_energy",
            label="Estimated energy [TeV]",
        )
        return hist.Hist(energy_axis, storage=hist.storage.Int64())

    def _process_observation(self, observation: Observation):
        events = observation.events
        on_events = events.select_region(self.on_region)
        self.h_on.fill(estimated_energy=on_events.energy.to_value("TeV"))

        if self.off_region:
            off_events = events.select_region(self.off_region)
            self.h_off.fill(estimated_energy=off_events.energy.to_value("TeV"))
        else:
            off_events = None

        return on_events, off_events

    def reset(self) -> None:
        """Remove all counts accumulated in both histograms."""
        self.h_on.reset()
        self.h_off.reset()

    def __call__(self, observations: Observations) -> tuple[hist.Hist, hist.Hist]:
        """Add observations and return the on and (unscaled) off histograms."""
        for observation in observations:
            self._process_observation(observation)

        return self.h_on, self.h_off
