"""Event-level ICRS sky offsets for background studies."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import angular_separation

if TYPE_CHECKING:
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


def _empty_angle() -> u.Quantity:
    """Return an independent empty offset array."""
    return np.empty(0, dtype=float) * u.deg


def _empty_energy() -> u.Quantity:
    """Return an independent empty energy array."""
    return np.empty(0, dtype=float) * u.TeV


@dataclass(eq=False)
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
        Effective observation time. Unitless values are interpreted as
        seconds.

    """

    x: Iterable[float] | u.Quantity = field(default_factory=_empty_angle)
    y: Iterable[float] | u.Quantity = field(default_factory=_empty_angle)
    energy: Iterable[float] | u.Quantity = field(default_factory=_empty_energy)
    livetime: float | u.Quantity = field(default_factory=lambda: 0 * u.s)

    def __post_init__(self) -> None:
        """Convert inputs to canonical units and validate the sample."""
        self.x = _as_quantity(self.x, u.deg, "x")
        self.y = _as_quantity(self.y, u.deg, "y")
        self.energy = _as_quantity(self.energy, u.TeV, "energy")

        if not (len(self.x) == len(self.y) == len(self.energy)):
            raise ValueError("x, y, and energy must contain the same number of events")
        if not all(np.all(np.isfinite(values.value)) for values in (self.x, self.y, self.energy)):
            raise ValueError("x, y, and energy must contain only finite values")

        try:
            if isinstance(self.livetime, u.Quantity):
                converted_livetime = self.livetime.to(u.s, copy=True)
            else:
                converted_livetime = np.asarray(self.livetime, dtype=float) * u.s
        except (TypeError, ValueError, u.UnitConversionError) as error:
            raise ValueError("livetime must be a scalar time") from error

        if converted_livetime.ndim != 0 or not np.isfinite(converted_livetime.value):
            raise ValueError("livetime must be a finite scalar time")
        if converted_livetime < 0 * u.s:
            raise ValueError("livetime must be non-negative")
        if self.energy.size > 0 and converted_livetime == 0 * u.s:
            raise ValueError("livetime must be positive when events are present")

        self.livetime = converted_livetime

    def __len__(self) -> int:
        """Return the number of stored events."""
        return len(self.energy)

    @property
    def radius(self) -> u.Quantity:
        """Exact angular separation of every event from its pointing."""
        return angular_separation(0 * u.deg, 0 * u.deg, self.x, self.y).to(u.deg)

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

        energy = self.energy.to_value(u.TeV)
        selected = (energy >= low) & (energy < high)
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
            x=np.concatenate((self.x.to_value(u.deg), other.x.to_value(u.deg))) * u.deg,
            y=np.concatenate((self.y.to_value(u.deg), other.y.to_value(u.deg))) * u.deg,
            energy=np.concatenate((self.energy.to_value(u.TeV), other.energy.to_value(u.TeV))) * u.TeV,
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
