"""Histograms of reconstructed events in telescope-centered coordinates."""

from collections.abc import Iterable

import astropy.units as u
import hist
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame


def _validate_edges(edges: Iterable[float], name: str, unit: u.Unit) -> np.ndarray:
    """Convert axis edges to the expected unit and validate them."""
    try:
        if isinstance(edges, u.Quantity):
            values = np.asarray(edges.to_value(unit), dtype=float)
        else:
            values = np.asarray(edges, dtype=float)
    except (TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must contain numeric values in {unit}") from error

    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must contain at least two one-dimensional bin edges")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")

    return values


class CameraImage:
    """Accumulate events in sky-offset and reconstructed-energy bins.

    Parameters
    ----------
    center
        Telescope pointing. Event directions are transformed to a
        :class:`~astropy.coordinates.SkyOffsetFrame` centered here.
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

    REQUIRED_COLUMNS = ("reco_alt", "reco_az", "reco_energy")

    def __init__(
        self,
        center: SkyCoord,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
    ) -> None:
        self.center = center
        self.telescope_frame = SkyOffsetFrame(origin=center)
        self.x_edges = _validate_edges(x_edges, "x_edges", u.deg)
        self.y_edges = _validate_edges(y_edges, "y_edges", u.deg)
        self.e_edges = _validate_edges(e_edges, "e_edges", u.TeV)

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

    def fill(self, events: pd.DataFrame) -> None:
        """Transform and add reconstructed events to the histogram.

        The ``reco_alt`` and ``reco_az`` columns are interpreted as degrees,
        and ``reco_energy`` is interpreted as TeV.
        """
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in events.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"events must contain the following columns: {missing}")

        directions = SkyCoord(
            alt=np.asarray(events["reco_alt"], dtype=float) * u.deg,
            az=np.asarray(events["reco_az"], dtype=float) * u.deg,
            frame=self.center.frame,
        )
        offsets = directions.transform_to(self.telescope_frame)

        self.histogram.fill(
            x=offsets.lon.to_value(u.deg),
            y=offsets.lat.to_value(u.deg),
            energy=np.asarray(events["reco_energy"], dtype=float),
        )
