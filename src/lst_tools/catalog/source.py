"""Uniform representation of a gamma-ray catalog source.

Each catalog is loaded into a tuple of :class:`CatalogSource`, which carries
the source name, the ICRS position and an optional angular extension
(``None`` for point-like sources).  This uniform representation lets the
plotting and filtering code treat every catalog the same way.
"""

from dataclasses import dataclass
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord


@dataclass(frozen=True)
class CatalogSource:
    """A single catalog entry: name, ICRS position and optional extension.

    ``extension`` is the characteristic angular radius of the source (e.g.
    the 39% containment radius for LHAASO, the Gaussian sigma for HESS, the
    model semi-major axis for Fermi).  ``None`` marks a point-like source.
    """

    name: str
    coord: SkyCoord
    catalog: str
    extension: Optional[u.Quantity] = None

    @property
    def is_extended(self) -> bool:
        """Whether the source has a well-defined, non-zero extension."""
        return self.extension is not None and self.extension > 0 * u.deg
