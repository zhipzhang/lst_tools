"""Event-level sky offsets and their observation metadata."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame
from numpy.typing import NDArray

from ..utils import altaz_to_icrs, mean_direction

_MINIMUM_TIME_STEP = 1 * u.s
_EARTH_ANGULAR_VELOCITY = 360 * u.deg / u.day
_ROTATION_SAMPLE_COUNT = 17
_MIN_COS_ALT = 1e-6


def _as_1d_array(values: Iterable[float] | u.Quantity, unit: u.Unit, name: str) -> NDArray[np.float64]:
    try:
        if isinstance(values, u.Quantity):
            array = values.to_value(unit)
        else:
            array = np.array(values, dtype=float)
    except (TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must contain numeric values in {unit}") from error

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _as_scalar_quantity(
    value: float | u.Quantity,
    unit: u.Unit,
    name: str,
    *,
    allow_unitless: bool = False,
) -> u.Quantity:
    try:
        quantity = u.Quantity(value, unit) if allow_unitless else value.to(unit, copy=True)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must be a quantity compatible with {unit}") from error

    if quantity.ndim != 0 or not np.isfinite(quantity.value):
        raise ValueError(f"{name} must be a finite scalar")
    return quantity


def _mean_pointing_icrs(fov_frame: FoVAltAzFrame, location: EarthLocation) -> SkyCoord:
    """Compute the spherical mean ICRS pointing across per-event AltAz origins."""
    origin = fov_frame.origin
    icrs_origins = altaz_to_icrs(origin.obstime, origin.alt, origin.az, location=location)
    return mean_direction(icrs_origins)


def _peak_fov_rotation_rate(
    pointing_icrs: SkyCoord,
    time_start: Time,
    duration_s: float,
    location: EarthLocation,
) -> u.Quantity:
    """Return the maximum instantaneous AltAz FoV rotation rate across an interval."""
    sample_times = time_start + np.linspace(0.0, duration_s, _ROTATION_SAMPLE_COUNT) * u.s
    sampled = pointing_icrs.transform_to(AltAz(obstime=sample_times, location=location))
    cos_alt = np.maximum(np.cos(sampled.alt.rad), _MIN_COS_ALT)
    rates = _EARTH_ANGULAR_VELOCITY * np.cos(location.lat.rad) * np.abs(np.cos(sampled.az.rad)) / cos_alt
    return rates.max()


def _compute_slice_count(
    pointing_icrs: SkyCoord,
    time_start: Time,
    duration_s: float,
    location: EarthLocation,
    max_fov_rotation: u.Quantity,
) -> int:
    """Choose enough equal-duration slices to keep FoV rotation per slice within tolerance."""
    if duration_s <= 0.0:
        return 1
    peak_rate = _peak_fov_rotation_rate(pointing_icrs, time_start, duration_s, location)
    if peak_rate <= 0 * (u.deg / u.s):
        return 1
    time_step_s = max(_MINIMUM_TIME_STEP.to_value(u.s), float((max_fov_rotation / peak_rate).to_value(u.s)))
    return max(1, int(np.ceil(duration_s / time_step_s - 1e-9)))


@dataclass(frozen=True)
class ObservationFrame:
    """Field-of-view frame and mean pointing metadata for one observation."""

    fov_frame: FoVICRSFrame | FoVAltAzFrame
    mean_time: Time
    pointing_alt: u.Quantity
    pointing_az: u.Quantity

    def __post_init__(self) -> None:
        if not isinstance(self.fov_frame, (FoVICRSFrame, FoVAltAzFrame)):
            raise TypeError("fov_frame must be a FoVICRSFrame or FoVAltAzFrame")
        if not isinstance(self.mean_time, Time) or not self.mean_time.isscalar:
            raise ValueError("mean_time must be a scalar Time")

        for name in ("pointing_alt", "pointing_az"):
            angle = _as_scalar_quantity(getattr(self, name), u.deg, name)
            object.__setattr__(self, name, angle)

    @classmethod
    def from_altaz_pointing(
        cls,
        pointing: SkyCoord,
        location: EarthLocation,
        obstime: Time,
    ) -> "ObservationFrame":
        """Build a scalar AltAz observation frame at one representative time and pointing."""
        return cls(
            fov_frame=FoVAltAzFrame(origin=pointing, location=location, obstime=obstime),
            mean_time=obstime,
            pointing_alt=pointing.alt,
            pointing_az=pointing.az,
        )


class SkyOffsetEvents:
    """One observation's events in telescope sky-offset coordinates.

    Unitless offsets are interpreted as degrees and unitless energies as TeV.
    The required frame identifies the observation; exclusion and combination
    of observations belong to the background-making layer.

    For GADF/Gammapy background IRFs, ``livetime`` should hold the elapsed
    observation duration (ontime, including dead time), matching Gammapy's
    background-map integration over ``ontime``.
    """

    def __init__(
        self,
        x: Iterable[float] | u.Quantity = (),
        y: Iterable[float] | u.Quantity = (),
        energy: Iterable[float] | u.Quantity = (),
        livetime: u.Quantity = 0 * u.s,
        *,
        frame: ObservationFrame,
    ) -> None:
        if not isinstance(frame, ObservationFrame):
            raise TypeError("frame must be an ObservationFrame")

        self.x: NDArray[np.float64] = _as_1d_array(x, u.deg, "x")
        self.y: NDArray[np.float64] = _as_1d_array(y, u.deg, "y")
        self.energy: NDArray[np.float64] = _as_1d_array(energy, u.TeV, "energy")
        self.livetime: u.Quantity = _as_scalar_quantity(livetime, u.s, "livetime")
        self.frame = frame

        if self.livetime < 0 * u.s:
            raise ValueError("livetime must be non-negative")
        if not (len(self.x) == len(self.y) == len(self.energy)):
            raise ValueError("x, y, and energy must contain the same number of events")
        if not all(np.all(np.isfinite(values)) for values in (self.x, self.y, self.energy)):
            raise ValueError("x, y, and energy must contain only finite values")
        if self.energy.size > 0 and self.livetime == 0 * u.s:
            raise ValueError("livetime must be positive when events are present")

    def __len__(self) -> int:
        return len(self.energy)

    @property
    def radius(self) -> NDArray[np.float64]:
        """Flat offset distance of every event from its pointing, in degrees."""
        return np.hypot(self.x, self.y)

    def select_energy(
        self,
        energy_low: float | u.Quantity,
        energy_high: float | u.Quantity,
    ) -> "SkyOffsetEvents":
        """Return events in ``[energy_low, energy_high)`` with unchanged metadata."""
        low = _as_scalar_quantity(energy_low, u.TeV, "energy_low", allow_unitless=True).to_value(u.TeV)
        high = _as_scalar_quantity(energy_high, u.TeV, "energy_high", allow_unitless=True).to_value(u.TeV)
        if low >= high:
            raise ValueError("energy_low must be less than energy_high")
        return self._subset((self.energy >= low) & (self.energy < high))

    def time_slices(self, max_fov_rotation: float | u.Quantity = 0.2 * u.deg) -> list["SkyOffsetEvents"]:
        """Split a tracked AltAz observation into equal-duration scalar-pointing slices."""
        rotation_limit = _as_scalar_quantity(max_fov_rotation, u.deg, "max_fov_rotation", allow_unitless=True)
        if rotation_limit <= 0 * u.deg:
            raise ValueError("max_fov_rotation must be positive")

        fov_frame = self._require_altaz_frame()
        if fov_frame.origin.isscalar:
            return [self]
        if fov_frame.location is None:
            raise ValueError("time_slices requires a FoVAltAzFrame with an observatory location")
        if len(fov_frame.origin) == 0:
            return [self._empty_scalar_slice(fov_frame.location)]

        times = fov_frame.origin.obstime
        time_start = times.min()
        duration_s = float((times.max() - time_start).to_value(u.s))
        pointing_icrs = _mean_pointing_icrs(fov_frame, fov_frame.location)
        n_slices = _compute_slice_count(pointing_icrs, time_start, duration_s, fov_frame.location, rotation_limit)
        return self._split_into_equal_slices(times, time_start, duration_s, pointing_icrs, fov_frame.location, n_slices)

    def _subset(
        self,
        mask: NDArray[np.bool_],
        *,
        livetime: u.Quantity | None = None,
        frame: ObservationFrame | None = None,
    ) -> "SkyOffsetEvents":
        return SkyOffsetEvents(
            x=self.x[mask],
            y=self.y[mask],
            energy=self.energy[mask],
            livetime=self.livetime if livetime is None else livetime,
            frame=self.frame if frame is None else frame,
        )

    def _require_altaz_frame(self) -> FoVAltAzFrame:
        fov_frame = self.frame.fov_frame
        if not isinstance(fov_frame, FoVAltAzFrame):
            raise TypeError("time_slices requires a FoVAltAzFrame")
        return fov_frame

    def _empty_scalar_slice(self, location: EarthLocation) -> "SkyOffsetEvents":
        mean_time = self.frame.mean_time
        pointing = SkyCoord(
            alt=self.frame.pointing_alt,
            az=self.frame.pointing_az,
            frame=AltAz(obstime=mean_time, location=location),
        )
        frame = ObservationFrame.from_altaz_pointing(pointing, location, mean_time)
        return SkyOffsetEvents(livetime=self.livetime, frame=frame)

    def _split_into_equal_slices(
        self,
        times: Time,
        time_start: Time,
        duration_s: float,
        pointing_icrs: SkyCoord,
        location: EarthLocation,
        n_slices: int,
    ) -> list["SkyOffsetEvents"]:
        edges = time_start + np.linspace(0.0, duration_s, n_slices + 1) * u.s
        mean_times = time_start + (np.arange(n_slices) + 0.5) * (duration_s / n_slices) * u.s
        pointings = pointing_icrs.transform_to(AltAz(obstime=mean_times, location=location))
        slice_livetime = self.livetime / n_slices

        slices: list[SkyOffsetEvents] = []
        for index, (start, stop) in enumerate(pairwise(edges)):
            in_window = (times >= start) & (times <= stop if index == n_slices - 1 else times < stop)
            frame = ObservationFrame.from_altaz_pointing(pointings[index], location, mean_times[index])
            slices.append(self._subset(in_window, livetime=slice_livetime, frame=frame))
        return slices
