"""Event-level reconstructed coordinates for camera-background studies."""

from collections.abc import Iterable

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from .camera import CameraImage


def _as_quantity(values: Iterable[float], unit: u.Unit, name: str) -> u.Quantity:
    """Return a copied, one-dimensional quantity in ``unit``."""
    try:
        if isinstance(values, u.Quantity):
            quantity = values.to(unit, copy=True)
        else:
            quantity = np.asarray(values, dtype=float) * unit
    except (TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must contain numeric values in {unit}") from error

    if quantity.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return quantity


def _energy_bound(value: float | u.Quantity, name: str) -> float:
    """Convert a scalar energy bound to TeV."""
    try:
        if isinstance(value, u.Quantity):
            bound = float(value.to_value(u.TeV))
        else:
            bound = float(value)
    except (TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must be a scalar energy in TeV") from error

    if not np.isfinite(bound):
        raise ValueError(f"{name} must be finite")
    return bound


class CameraEvents:
    """Retain event-level sky offsets and reconstructed energies.

    Unlike :class:`~lst_tools.bkg.camera.CameraImage`, this class does not bin
    its data. The same events can therefore be selected or converted into
    multiple images using different bin edges.

    Parameters
    ----------
    center
        Telescope pointing used as the sky-offset origin.
    x, y
        Event sky-offset longitude and latitude. Unitless values are treated
        as degrees.
    energy
        Event reconstructed energies. Unitless values are treated as TeV.
    """

    REQUIRED_COLUMNS = ("reco_alt", "reco_az", "reco_energy")

    def __init__(
        self,
        center: SkyCoord,
        x: Iterable[float],
        y: Iterable[float],
        energy: Iterable[float],
    ) -> None:
        self.center = center
        self.telescope_frame = SkyOffsetFrame(origin=center)
        self.x = _as_quantity(x, u.deg, "x")
        self.y = _as_quantity(y, u.deg, "y")
        self.energy = _as_quantity(energy, u.TeV, "energy")

        if not (len(self.x) == len(self.y) == len(self.energy)):
            raise ValueError("x, y, and energy must contain the same number of events")

    @classmethod
    def from_dataframe(cls, events: pd.DataFrame, center: SkyCoord) -> "CameraEvents":
        """Transform reconstructed Alt/Az columns into sky offsets.

        Altitude and azimuth values are interpreted as degrees, while energy
        values are interpreted as TeV. The input DataFrame is not modified.
        """
        missing_columns = [column for column in cls.REQUIRED_COLUMNS if column not in events.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"events must contain the following columns: {missing}")

        directions = SkyCoord(
            alt=np.asarray(events["reco_alt"], dtype=float) * u.deg,
            az=np.asarray(events["reco_az"], dtype=float) * u.deg,
            frame=center.frame,
        )
        offsets = directions.transform_to(SkyOffsetFrame(origin=center))

        return cls(
            center=center,
            x=offsets.lon,
            y=offsets.lat,
            energy=np.asarray(events["reco_energy"], dtype=float) * u.TeV,
        )

    def __len__(self) -> int:
        return len(self.energy)

    @property
    def radius(self) -> u.Quantity:
        """Angular radius of every event in the sky-offset plane."""
        return np.hypot(self.x, self.y)

    def select_energy(
        self,
        energy_low: float | u.Quantity,
        energy_high: float | u.Quantity,
    ) -> "CameraEvents":
        """Return events in the lower-inclusive interval ``[low, high)``."""
        low = _energy_bound(energy_low, "energy_low")
        high = _energy_bound(energy_high, "energy_high")
        if low >= high:
            raise ValueError("energy_low must be less than energy_high")

        energy = self.energy.to_value(u.TeV)
        selected = (energy >= low) & (energy < high)
        return CameraEvents(
            center=self.center,
            x=self.x[selected],
            y=self.y[selected],
            energy=self.energy[selected],
        )

    def to_image(
        self,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
    ) -> CameraImage:
        """Bin these events into a new, independent camera image."""
        image = CameraImage(
            center=self.center,
            x_edges=x_edges,
            y_edges=y_edges,
            e_edges=e_edges,
        )
        image.histogram.fill(
            x=self.x.to_value(u.deg),
            y=self.y.to_value(u.deg),
            energy=self.energy.to_value(u.TeV),
        )
        return image
