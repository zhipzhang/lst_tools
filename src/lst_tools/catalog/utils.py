"""Utilities for comparing run pointings with catalog sources."""

from collections.abc import Iterable

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .source import CatalogSource


def select_runs_away_from_sources(
    statistics: pd.DataFrame,
    sources: Iterable[CatalogSource],
    *,
    min_separation: u.Quantity = 3 * u.deg,
    extension_factor: float = 2.5,
) -> pd.DataFrame:
    """Return runs whose pointings are sufficiently far from every source.

    This function applies only catalog-separation selection. Callers should
    apply any DataCheck quality cuts before passing ``statistics`` here.

    For an extended source, the exclusion radius is ``min_separation`` plus
    ``extension_factor`` times the source extension. Point-like sources use
    only ``min_separation``.
    """
    required_columns = {"mean_ra", "mean_dec"}
    missing_columns = required_columns.difference(statistics.columns)
    if missing_columns:
        raise ValueError(f"statistics is missing pointing columns: {sorted(missing_columns)}")

    min_separation_deg = u.Quantity(min_separation).to_value(u.deg)
    if not np.isfinite(min_separation_deg) or min_separation_deg < 0:
        raise ValueError("min_separation must be finite and non-negative")
    if not np.isfinite(extension_factor) or extension_factor < 0:
        raise ValueError("extension_factor must be finite and non-negative")

    pointings = SkyCoord(
        ra=statistics["mean_ra"].to_numpy() * u.deg,
        dec=statistics["mean_dec"].to_numpy() * u.deg,
    )
    keep = np.ones(len(statistics), dtype=bool)

    for source in sources:
        extension_deg = source.extension.to_value(u.deg) if source.is_extended else 0.0
        exclusion_radius = min_separation_deg + extension_factor * extension_deg
        keep &= pointings.separation(source.coord).to_value(u.deg) > exclusion_radius

    return statistics.loc[keep].copy()
