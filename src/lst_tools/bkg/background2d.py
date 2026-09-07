"""Construct radial Gammapy background models from sky-offset events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from gammapy.irf import Background2D
from gammapy.maps import MapAxis

from .events import SkyOffsetEvents
from .utils import validate_edges


@dataclass
class Background2DMaker:
    """Reusable configuration for constructing radial background-rate models.

    Parameters
    ----------
    energy_edges
        Reconstructed-energy bin edges. Unitless values are interpreted as
        TeV.
    theta_edges
        Offset-radius bin edges. Unitless values are interpreted as degrees.
    spectral_index
        Power-law index used to convert each integral energy-bin rate into a
        differential rate at the logarithmic bin center.

    """

    energy_edges: Iterable[float] | u.Quantity
    theta_edges: Iterable[float] | u.Quantity
    spectral_index: float = -2.0

    def __post_init__(self) -> None:
        """Validate and normalize the binning configuration."""
        try:
            spectral_index = float(self.spectral_index)
        except (TypeError, ValueError) as error:
            raise ValueError("spectral_index must be a finite number") from error
        if not np.isfinite(spectral_index):
            raise ValueError("spectral_index must be a finite number")

        energy_values = validate_edges(self.energy_edges, "energy_edges", u.TeV)
        if energy_values[0] <= 0:
            raise ValueError("energy_edges must be positive")
        theta_values = validate_edges(self.theta_edges, "theta_edges", u.deg)
        if theta_values[0] < 0 or theta_values[-1] > 180:
            raise ValueError("theta_edges must be between 0 and 180 degrees")

        self.energy_edges = energy_values * u.TeV
        self.theta_edges = theta_values * u.deg
        self.spectral_index = spectral_index

    @staticmethod
    def _validate_events(events: SkyOffsetEvents) -> None:
        """Validate an event sample before constructing a physical rate."""
        if not isinstance(events, SkyOffsetEvents):
            raise TypeError("events must be a SkyOffsetEvents object")
        if events.livetime <= 0 * u.s:
            raise ValueError("events livetime must be positive")

    def counts(self, events: SkyOffsetEvents) -> np.ndarray:
        """Histogram events in reconstructed energy and offset radius."""
        self._validate_events(events)
        counts, _, _ = np.histogram2d(
            events.energy.to_value(u.TeV),
            events.radius.to_value(u.deg),
            bins=(self.energy_edges.to_value(u.TeV), self.theta_edges.to_value(u.deg)),
        )
        return counts.astype(np.int64)

    @property
    def _annular_solid_angle(self) -> u.Quantity:
        """Exact solid angle of every offset annulus."""
        theta = self.theta_edges.to_value(u.rad)
        return np.diff(2 * np.pi * (1 - np.cos(theta))) * u.sr

    @property
    def _effective_energy_width(self) -> u.Quantity:
        """Power-law-weighted energy widths at logarithmic bin centers."""
        low = self.energy_edges[:-1].to(u.MeV)
        high = self.energy_edges[1:].to(u.MeV)
        reference = np.sqrt(low * high)
        exponent = self.spectral_index + 1

        if self.spectral_index == -1:
            width = reference * np.log(high / low)
        else:
            high_ratio = (high / reference).to_value(u.one)
            low_ratio = (low / reference).to_value(u.one)
            width = reference * (high_ratio**exponent - low_ratio**exponent) / exponent

        if np.any(~np.isfinite(width.value)) or np.any(width <= 0 * u.MeV):
            raise ValueError("spectral_index produces invalid effective energy widths")
        return width.to(u.MeV)

    def differential_rate(self, events: SkyOffsetEvents) -> u.Quantity:
        """Normalize counts by livetime, solid angle, and effective width."""
        counts = self.counts(events)
        normalization = (
            events.livetime
            * self._effective_energy_width[:, np.newaxis]
            * self._annular_solid_angle[np.newaxis, :]
        )
        return (counts / normalization).to(1 / (u.MeV * u.s * u.sr))

    def __call__(self, events: SkyOffsetEvents) -> Background2D:
        """Construct a Gammapy background model from one combined sample."""
        rate = self.differential_rate(events)
        energy_axis = MapAxis.from_edges(self.energy_edges, name="energy", interp="log")
        offset_axis = MapAxis.from_edges(self.theta_edges, name="offset")
        return Background2D(
            axes=[energy_axis, offset_axis],
            data=rate.value,
            unit=rate.unit,
            meta={
                "LIVETIME": events.livetime.to_value(u.s),
                "SPEC_IDX": self.spectral_index,
            },
        )
