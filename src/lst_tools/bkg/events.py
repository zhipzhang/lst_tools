"""Event-level ICRS sky offsets and their observation metadata."""

from collections.abc import Iterable
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.time import Time
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame
from numpy.typing import NDArray


def _as_array(values: Iterable[float] | u.Quantity, unit: u.Unit, name: str) -> NDArray[np.float64]:
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


def _energy_bound(value: float | u.Quantity, name: str) -> float:
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
            value = getattr(self, name)
            try:
                angle = value.to(u.deg)
            except (AttributeError, u.UnitConversionError) as error:
                raise ValueError(f"{name} must be an angle quantity") from error
            if angle.ndim != 0 or not np.isfinite(angle.value):
                raise ValueError(f"{name} must be a finite scalar")
            object.__setattr__(self, name, angle)


class SkyOffsetEvents:
    """One observation's events in telescope sky-offset coordinates.

    Unitless offsets are interpreted as degrees and unitless energies as TeV.
    The required frame identifies the observation; exclusion and combination
    of observations belong to the background-making layer.
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
        self.x: NDArray[np.float64] = _as_array(x, u.deg, "x")
        self.y: NDArray[np.float64] = _as_array(y, u.deg, "y")
        self.energy: NDArray[np.float64] = _as_array(energy, u.TeV, "energy")
        self.frame = frame

        if not isinstance(frame, ObservationFrame):
            raise TypeError("frame must be an ObservationFrame")
        try:
            self.livetime: u.Quantity = livetime.to(u.s, copy=True)
        except (AttributeError, u.UnitConversionError) as error:
            raise ValueError("livetime must be a time quantity") from error

        if self.livetime.ndim != 0 or not np.isfinite(self.livetime.value):
            raise ValueError("livetime must be a finite scalar")
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
        low = _energy_bound(energy_low, "energy_low")
        high = _energy_bound(energy_high, "energy_high")
        if low >= high:
            raise ValueError("energy_low must be less than energy_high")

        selected = (self.energy >= low) & (self.energy < high)
        return SkyOffsetEvents(
            x=self.x[selected],
            y=self.y[selected],
            energy=self.energy[selected],
            livetime=self.livetime,
            frame=self.frame,
        )
