"""Loaders that turn gammapy catalogs into tuples of :class:`CatalogSource`.

Each catalog is loaded once (lazily and cached with ``lru_cache``).
``select_region`` filters any source tuple to a circular sky region.

Supported catalogs: Fermi-LAT 4FGL, Gamma-Cat, LHAASO 1LHAASO (KM2A/WCDA),
HAWC 3HWC, HESS HGPS.
"""

from functools import lru_cache
from typing import Any, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.catalog import (
    SourceCatalog1LHAASO,
    SourceCatalog3HWC,
    SourceCatalog4FGL,
    SourceCatalogGammaCat,
    SourceCatalogHGPS,
)

from .source import CatalogSource


def _deg(value: Any) -> float:
    """Convert a plain number or a Quantity to decimal degrees."""
    return float(u.Quantity(value, u.deg).to_value(u.deg))


@lru_cache(maxsize=1)
def load_fermi_sources() -> Tuple[CatalogSource, ...]:
    """Load the Fermi-LAT 4FGL catalog, attaching extensions when known."""
    catalog = SourceCatalog4FGL()

    # The extended-sources table is keyed by the common source name
    # (e.g. "IC 443"), which the main table exposes as ``Extended_Source_Name``.
    extensions = {}
    for row in catalog.extended_sources_table:
        semi_major = row["Model_SemiMajor"]
        if np.isfinite(semi_major) and semi_major > 0:
            extensions[str(row["Source_Name"]).strip()] = float(semi_major) * u.deg

    sources = []
    for row in catalog.table:
        ext_name = str(row["Extended_Source_Name"]).strip()
        sources.append(
            CatalogSource(
                name=str(row["Source_Name"]),
                coord=SkyCoord(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), unit="deg"),
                catalog="fermi",
                extension=extensions.get(ext_name),
            )
        )
    return tuple(sources)


@lru_cache(maxsize=1)
def load_lhaaso_sources() -> Tuple[CatalogSource, ...]:
    """Load the LHAASO 1LHAASO catalog, split into KM2A and WCDA detections.

    Sources detected independently by both arrays appear twice, once per
    component, with ``_km2a``/``_wcda`` appended to the name.  The extension
    is the 39% containment radius ``r39``.
    """
    catalog = SourceCatalog1LHAASO()
    sources = []
    for source in catalog:
        data = source.data
        components = (
            ("a", data["RAJ2000"], data["DECJ2000"], data["r39"]),
            ("b", data["RAJ2000_b"], data["DECJ2000_b"], data["r39_b"]),
        )
        for component, ra, dec, r39 in components:
            model = data[f"Model_{component}"].strip()
            if model not in ("KM2A", "WCDA"):
                continue
            r39_deg = _deg(r39)
            sources.append(
                CatalogSource(
                    name=f"{data['Source_Name']}_{model.lower()}",
                    coord=SkyCoord(ra=_deg(ra), dec=_deg(dec), unit="deg"),
                    catalog=f"lhaaso_{model.lower()}",
                    extension=r39_deg * u.deg if np.isfinite(r39_deg) else None,
                )
            )
    return tuple(sources)


@lru_cache(maxsize=1)
def load_hawc_sources() -> Tuple[CatalogSource, ...]:
    """Load the HAWC 3HWC catalog (all sources treated as point-like)."""
    table = SourceCatalog3HWC().table
    return tuple(
        CatalogSource(
            name=str(row["source_name"]),
            coord=SkyCoord(ra=float(row["ra"]), dec=float(row["dec"]), unit="deg"),
            catalog="hawc",
        )
        for row in table
    )


@lru_cache(maxsize=1)
def load_gammacat_sources() -> Tuple[CatalogSource, ...]:
    """Load the Gamma-Cat catalog; extension is the morphological sigma (``morph_sigma``)."""
    table = SourceCatalogGammaCat().table
    sources = []
    for row in table:
        size = row["morph_sigma"]
        size2 = row["morph_sigma2"]
        ext_candidates = []
        for s in (size, size2):
            if not np.ma.is_masked(s):
                s_deg = float(s)
                if np.isfinite(s_deg) and s_deg > 0:
                    ext_candidates.append(s_deg)
        extension = max(ext_candidates) * u.deg if ext_candidates else None
        sources.append(
            CatalogSource(
                name=str(row["common_name"]),
                coord=SkyCoord(ra=float(row["ra"]), dec=float(row["dec"]), unit="deg"),
                catalog="gammacat",
                extension=extension,
            )
        )
    return tuple(sources)


load_gamma_cat_sources = load_gammacat_sources


@lru_cache(maxsize=1)
def load_hess_sources() -> Tuple[CatalogSource, ...]:
    """Load the HESS HGPS catalog; the extension is the Gaussian ``Size``."""
    table = SourceCatalogHGPS().table
    sources = []
    for row in table:
        size = row["Size"]
        sources.append(
            CatalogSource(
                name=str(row["Source_Name"]),
                coord=SkyCoord(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), unit="deg"),
                catalog="hess",
                extension=float(size) * u.deg if np.isfinite(size) else None,
            )
        )
    return tuple(sources)


@lru_cache(maxsize=1)
def load_tev_sources() -> Tuple[CatalogSource, ...]:
    """Load every TeV catalog used to reject source-contaminated pointings.

    Concatenates HESS, LHAASO, HAWC and Gamma-Cat, so a source detected by
    several catalogs appears once per catalog.
    """
    return load_hess_sources() + load_lhaaso_sources() + load_hawc_sources() + load_gammacat_sources()


def select_region(sources: Tuple[CatalogSource, ...], center: SkyCoord, radius: u.Quantity) -> List[CatalogSource]:
    """Return the sources within ``radius`` of ``center``."""
    if len(sources) == 0:
        return []
    coords = SkyCoord([s.coord.icrs for s in sources])
    separations = coords.separation(center)
    return [s for s, sep in zip(sources, separations) if sep <= radius]
