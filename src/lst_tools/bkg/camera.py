"""Count histograms for individual observations in sky-offset coordinates."""

from collections.abc import Iterable

import astropy.units as u
import hist
import numpy as np

from .events import ObservationFrame, SkyOffsetEvents
from .utils import validate_edges


class CameraImage:
    """Counts, livetime, and frame metadata for one observation."""

    def __init__(
        self,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
        *,
        frame: ObservationFrame,
        live_time: u.Quantity = 0 * u.s,
    ) -> None:
        self.x_edges = validate_edges(x_edges, "x_edges", u.deg)
        self.y_edges = validate_edges(y_edges, "y_edges", u.deg)
        self.e_edges = validate_edges(e_edges, "e_edges", u.TeV)
        if not isinstance(frame, ObservationFrame):
            raise TypeError("frame must be an ObservationFrame")
        self.frame = frame

        try:
            self.live_time = live_time.to(u.s, copy=True)
        except (AttributeError, u.UnitConversionError) as error:
            raise ValueError("live_time must be a time quantity") from error
        if self.live_time.ndim != 0 or not np.isfinite(self.live_time.value) or self.live_time < 0 * u.s:
            raise ValueError("live_time must be a finite non-negative scalar")

        self.histogram = hist.Hist(
            self._make_axis(self.x_edges, "x"),
            self._make_axis(self.y_edges, "y"),
            self._make_axis(self.e_edges, "energy"),
        )

    @classmethod
    def from_events(
        cls,
        events: SkyOffsetEvents,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
    ) -> "CameraImage":
        """Bin exactly one event sample without combining observation frames."""
        if not isinstance(events, SkyOffsetEvents):
            raise TypeError("events must be a SkyOffsetEvents object")
        image = cls(x_edges, y_edges, e_edges, frame=events.frame, live_time=events.livetime)
        image.histogram.fill(x=events.x, y=events.y, energy=events.energy)
        return image

    @staticmethod
    def _make_axis(edges, name):
        return hist.axis.Variable(edges, name=name, underflow=False, overflow=False)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Return uniform spatial pixel dimensions (dx, dy) in degrees."""
        dx = np.diff(self.x_edges)
        dy = np.diff(self.y_edges)
        if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
            raise ValueError("Uniform spatial binning is required.")
        return float(dx[0]), float(dy[0])

    @property
    def pixel_solid_angle(self) -> np.ndarray:
        """Return each count pixel's longitude-latitude area in steradians."""
        lon_width = np.diff(np.deg2rad(self.x_edges))
        lat_width = np.diff(np.deg2rad(self.y_edges))
        return lon_width[:, None] * lat_width[None, :]

    def to_radial(self, theta_edges: Iterable[float]) -> hist.Hist:
        """Bin counts into energy and offset-radius axes using pixel centers."""
        edges = validate_edges(theta_edges, "theta_edges", u.deg)
        x_centers = self.histogram.axes[0].centers
        y_centers = self.histogram.axes[1].centers
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        radius = np.hypot(xx, yy)
        counts = self.histogram.values()
        radial_counts = np.stack(
            [np.histogram(radius, bins=edges, weights=counts[:, :, i])[0] for i in range(counts.shape[2])]
        )

        result = hist.Hist(
            self._make_axis(self.e_edges, "energy"),
            self._make_axis(edges, "theta"),
        )
        result.values()[:] = radial_counts
        return result
