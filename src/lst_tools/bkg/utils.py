"""Shared validation helpers for background-model axes."""

from collections.abc import Iterable

import astropy.units as u
import numpy as np


def validate_edges(edges: Iterable[float], name: str, unit: u.Unit) -> np.ndarray:
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
