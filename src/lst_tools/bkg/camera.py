"""Histograms of reconstructed events in telescope-centered coordinates."""

from collections.abc import Iterable

import astropy.units as u
import hist

from .events import SkyOffsetEvents
from .utils import validate_edges


class CameraImage:
    """Accumulate events in sky-offset and reconstructed-energy bins.

    Parameters
    ----------
    x_edges, y_edges
        Spatial bin edges. Unitless values are interpreted as degrees.
    e_edges
        Reconstructed-energy bin edges. Unitless values are interpreted as
        TeV.

    Notes
    -----
    Calling :meth:`fill` adds to the existing histogram. Events outside any
    configured axis are discarded.
    """

    def __init__(
        self,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
    ) -> None:
        self.x_edges = validate_edges(x_edges, "x_edges", u.deg)
        self.y_edges = validate_edges(y_edges, "y_edges", u.deg)
        self.e_edges = validate_edges(e_edges, "e_edges", u.TeV)

        self.histogram = hist.Hist(
            hist.axis.Variable(
                self.x_edges,
                name="x",
                label="Sky-offset longitude [deg]",
                underflow=False,
                overflow=False,
            ),
            hist.axis.Variable(
                self.y_edges,
                name="y",
                label="Sky-offset latitude [deg]",
                underflow=False,
                overflow=False,
            ),
            hist.axis.Variable(
                self.e_edges,
                name="energy",
                label="Reconstructed energy [TeV]",
                underflow=False,
                overflow=False,
            ),
        )

    def fill(self, events: SkyOffsetEvents) -> None:
        """Add already transformed sky-offset events to the histogram."""
        if not isinstance(events, SkyOffsetEvents):
            raise TypeError("events must be a SkyOffsetEvents object")
        self.histogram.fill(
            x=events.x.to_value(u.deg),
            y=events.y.to_value(u.deg),
            energy=events.energy.to_value(u.TeV),
        )
