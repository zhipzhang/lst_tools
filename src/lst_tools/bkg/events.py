"""Event-level ICRS sky offsets for background studies."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .camera import CameraImage


def _as_array(values: Iterable[float] | u.Quantity, unit: u.Unit, name: str) -> NDArray[np.float64]:
    """Return a one-dimensional float array expressed in ``unit``."""
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


class SkyOffsetEvents:
    """Center-free event samples in telescope sky-offset coordinates.

    This is the public interface for the reduced event representation used by
    background makers. Coordinate transformation from an instrument-specific
    event format belongs in a separate adapter; instances therefore contain
    only offsets, reconstructed energies, and the effective livetime of the
    sample.

    Parameters
    ----------
    x, y
        ICRS-aligned offset longitude and latitude. Unitless values are
        interpreted as degrees.
    energy
        Reconstructed event energies. Unitless values are interpreted as TeV.
    livetime
        Effective observation time as a quantity, stored in seconds.

    Attributes
    ----------
    x, y : NDArray[np.float64]
        Offsets stored in degrees.
    energy : NDArray[np.float64]
        Energies stored in TeV.
    livetime : u.Quantity
        Effective observation time in seconds.

    """

    def __init__(
        self,
        x: Iterable[float] | u.Quantity = (),
        y: Iterable[float] | u.Quantity = (),
        energy: Iterable[float] | u.Quantity = (),
        livetime: u.Quantity = 0 * u.s,
    ) -> None:
        self.x: NDArray[np.float64] = _as_array(x, u.Unit("deg"), "x")
        self.y: NDArray[np.float64] = _as_array(y, u.Unit("deg"), "y")
        self.energy: NDArray[np.float64] = _as_array(energy, u.Unit("TeV"), "energy")

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
        """Return the number of stored events."""
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
        """Return events in ``[energy_low, energy_high)`` with unchanged livetime."""
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
        )

    def add(self, other: "SkyOffsetEvents") -> "SkyOffsetEvents":
        """Return a new sample with concatenated events and summed livetime."""
        if not isinstance(other, SkyOffsetEvents):
            raise TypeError("other must be a SkyOffsetEvents object")

        return SkyOffsetEvents(
            x=np.concatenate((self.x, other.x)),
            y=np.concatenate((self.y, other.y)),
            energy=np.concatenate((self.energy, other.energy)),
            livetime=self.livetime + other.livetime,
        )

    def __add__(self, other: "SkyOffsetEvents") -> "SkyOffsetEvents":
        """Return ``self.add(other)``."""
        return self.add(other)

    def to_image(
        self,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
    ) -> "CameraImage":
        """Bin these events into a new, independent camera image."""
        from .camera import CameraImage

        image = CameraImage(
            x_edges=x_edges,
            y_edges=y_edges,
            e_edges=e_edges,
        )
        image.fill(self)
        return image
