"""Event-level ICRS sky offsets for background studies."""

from collections.abc import Iterable

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, SkyOffsetFrame, angular_separation
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time

from .camera import CameraImage

LST_LOCATION = EarthLocation(
    lat=28.761758 * u.deg,
    lon=-17.890659 * u.deg,
    height=2200 * u.m,
)


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


def _mean_direction(coordinates: SkyCoord) -> SkyCoord:
    """Return the spherical mean of a set of ICRS directions."""
    mean_x, mean_y, mean_z = coordinates.cartesian.xyz.mean(axis=1)
    return SkyCoord(
        x=mean_x,
        y=mean_y,
        z=mean_z,
        representation_type="cartesian",
        frame="icrs",
    )


class CameraEvents:
    """Store reconstructed events as ICRS-aligned sky offsets.

    Unlike :class:`~lst_tools.bkg.camera.CameraImage`, this class does not bin
    its data. The same events can therefore be selected or converted into
    multiple images using different bin edges.

    These offsets are aligned with the celestial ICRS frame; they are not
    coordinates fixed to the physical camera. This distinction matters for
    two-dimensional studies because the camera rotates relative to the sky.

    Parameters
    ----------
    center
        Scalar sky position used as the offset origin.
    x, y
        ICRS-aligned offset longitude and latitude. Unitless values are
        treated as degrees.
    energy
        Event reconstructed energies. Unitless values are treated as TeV.
    """

    REQUIRED_COLUMNS = ("reco_alt", "reco_az", "reco_energy", "alt_tel", "az_tel", "trigger_time")

    def __init__(
        self,
        center: SkyCoord,
        x: Iterable[float],
        y: Iterable[float],
        energy: Iterable[float],
    ) -> None:
        if not center.isscalar:
            raise ValueError("center must be a scalar SkyCoord")

        self.center = center
        self.offset_frame = SkyOffsetFrame(origin=center)
        self.x = _as_quantity(x, u.deg, "x")
        self.y = _as_quantity(y, u.deg, "y")
        self.energy = _as_quantity(energy, u.TeV, "energy")

        if not (len(self.x) == len(self.y) == len(self.energy)):
            raise ValueError("x, y, and energy must contain the same number of events")

    @classmethod
    def from_dataframe(cls, events: pd.DataFrame, location: EarthLocation = LST_LOCATION) -> "CameraEvents":
        """Transform DL2 Alt/Az coordinates into ICRS-aligned sky offsets.

        Altitude and azimuth columns are interpreted as radians, trigger times
        as Unix seconds, and reconstructed energies as TeV. The offset origin
        is the spherical mean of the event-wise telescope pointings. The input
        DataFrame is not modified.
        """
        missing_columns = [column for column in cls.REQUIRED_COLUMNS if column not in events.columns]
        if missing_columns:
            raise ValueError(f"events is missing required columns: {', '.join(missing_columns)}")
        if events.empty:
            raise ValueError("events must contain at least one event")

        event_time = Time(events["trigger_time"].to_numpy(), format="unix")
        pointing_alt = events["alt_tel"].to_numpy(dtype=float) * u.rad
        pointing_az = events["az_tel"].to_numpy(dtype=float) * u.rad
        reco_alt = events["reco_alt"].to_numpy(dtype=float) * u.rad
        reco_az = events["reco_az"].to_numpy(dtype=float) * u.rad
        energy = events["reco_energy"].to_numpy(dtype=float) * u.TeV
        altaz_frame = AltAz(obstime=event_time, location=location)

        with erfa_astrom.set(ErfaAstromInterpolator(time_resolution=100 * u.s)):
            pointing = SkyCoord(
                alt=pointing_alt,
                az=pointing_az,
                frame=altaz_frame,
            ).icrs
            reconstructed = SkyCoord(
                alt=reco_alt,
                az=reco_az,
                frame=altaz_frame,
            ).icrs

        center = _mean_direction(pointing)
        offsets = reconstructed.transform_to(SkyOffsetFrame(origin=center))

        return cls(
            center=center,
            x=offsets.lon,
            y=offsets.lat,
            energy=energy,
        )

    def __len__(self) -> int:
        return len(self.energy)

    @property
    def radius(self) -> u.Quantity:
        """Exact angular separation of every event from the pointing center."""
        return angular_separation(0 * u.deg, 0 * u.deg, self.x, self.y).to(u.deg)

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
