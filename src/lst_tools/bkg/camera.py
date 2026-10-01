"""Histograms of reconstructed events in telescope-centered coordinates."""

from collections.abc import Iterable

import astropy.units as u
import hist
import numpy as np

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
        self.live_time = 0 * u.Unit("s")

    def fill(self, events: SkyOffsetEvents) -> None:
        """Add already transformed sky-offset events to the histogram."""
        if not isinstance(events, SkyOffsetEvents):
            raise TypeError("events must be a SkyOffsetEvents object")

        self.live_time += events.livetime
        self.histogram.fill(
            x=events.x,
            y=events.y,
            energy=events.energy,
        )

    def to_radial(self, theta_edges: Iterable[float]) -> hist.Hist:
        """Rebin the image into offset-radius and reconstructed-energy bins.

        Each x-y bin contributes all its counts to the radial bin containing
        the flat offset distance of its center from the pointing. Unitless
        ``theta_edges`` are interpreted as degrees.
        """
        edges = validate_edges(theta_edges, "theta_edges", u.deg)

        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
        radii = np.hypot(grid_x, grid_y)

        counts = self.histogram.values()
        radial_counts = np.stack(
            [np.histogram(radii, bins=edges, weights=counts[:, :, index])[0] for index in range(counts.shape[2])]
        )

        radial = hist.Hist(
            hist.axis.Variable(
                self.e_edges,
                name="energy",
                label="Reconstructed energy [TeV]",
                underflow=False,
                overflow=False,
            ),
            hist.axis.Variable(
                edges,
                name="theta",
                label="Offset radius [deg]",
                underflow=False,
                overflow=False,
            ),
        )
        radial.values()[:] = radial_counts
        return radial
