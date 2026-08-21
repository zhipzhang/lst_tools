"""Create reconstructed-energy count histograms for on/off regions."""

import astropy.units as u
import hist
import numpy as np
from gammapy.data import Observation, Observations
from gammapy.makers import WobbleRegionsFinder
from gammapy.maps import MapAxis
from regions import CircleSkyRegion


class OnOffCountsHistogram:
    """Accumulate on/off event counts as a function of energy and theta squared.

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Signal region.
    off_region_finder : `~gammapy.makers.WobbleRegionsFinder` or None
        Finder used to generate background-control regions for each
        observation. If None, only the on histogram is filled.
    energy_axis : `~gammapy.maps.MapAxis` or `~astropy.units.Quantity`
        Energy-bin edges. Unitless input is interpreted as TeV.

    Notes
    -----
    Calling an instance adds the supplied observations to the existing
    histograms. Use :meth:`reset` before a call when a fresh result is needed.
    The theta-squared axis has 1,000 linear bins between zero and the squared
    radius of the on region, expressed in degrees squared.
    """

    def __init__(
        self,
        on_region: CircleSkyRegion,
        off_region_finder: WobbleRegionsFinder | None,
        energy_axis: MapAxis | u.Quantity,
    ):
        self.on_region = on_region
        self.off_region_finder = off_region_finder

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

        radius = self.on_region.radius.to_value("deg")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("on_region radius must be finite and greater than zero")
        self.theta_squared_max = radius**2

        self.n_off = off_region_finder.n_off_regions if off_region_finder is not None else 0
        self.alpha = 1.0 / self.n_off if self.n_off > 0 else 0.0

        self.h_on = self._make_histogram()
        self.h_off = self._make_histogram()

    def _make_histogram(self) -> hist.Hist:
        energy_axis = hist.axis.Variable(
            self.energy_bins,
            name="estimated_energy",
            label="Estimated energy [TeV]",
        )
        theta_squared_axis = hist.axis.Regular(
            1_000,
            0,
            self.theta_squared_max,
            name="theta_squared",
            label=r"$\theta^2$ [deg$^2$]",
        )
        return hist.Hist(energy_axis, theta_squared_axis, storage=hist.storage.Int64())

    def _process_observation(self, observation: Observation):
        events = observation.events
        off_regions = []
        if self.off_region_finder is not None:
            pointing = observation.pointing.fixed_icrs
            off_regions, _ = self.off_region_finder.run(region=self.on_region, center=pointing)

        regions = [self.on_region, *off_regions]
        selected_events = events.select_region(regions)

        theta_squared_by_region = np.asarray(
            [region.center.separation(selected_events.radec).to_value("deg") ** 2 for region in regions]
        )
        nearest_region = np.argmin(theta_squared_by_region, axis=0)
        theta_squared = np.take_along_axis(
            theta_squared_by_region,
            nearest_region[np.newaxis, :],
            axis=0,
        )[0]

        on_mask = nearest_region == 0
        on_events = selected_events.select_row_subset(on_mask)
        self.h_on.fill(
            estimated_energy=on_events.energy.to_value("TeV"),
            theta_squared=theta_squared[on_mask],
        )

        if off_regions:
            off_mask = ~on_mask
            off_events = selected_events.select_row_subset(off_mask)
            self.h_off.fill(
                estimated_energy=off_events.energy.to_value("TeV"),
                theta_squared=theta_squared[off_mask],
            )
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


# Backward-compatible aliases for the previous class names.
HistogramCountsOnOff = OnOffCountsHistogram
HistogramCountsOnoff = OnOffCountsHistogram
