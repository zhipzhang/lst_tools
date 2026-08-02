"""
Helper functions for plotting catalog data.

Support catalog:
    - Fermi-LAT 4FGL
    - LHAASO 1LHAASO
    - HESS HESS-Catalog
"""

from typing import Any, Dict, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.catalog import (
    SourceCatalog1LHAASO,
    SourceCatalog3HWC,
    SourceCatalog4FGL,
    SourceCatalogHGPS,
    SourceCatalogObject1LHAASO,
)

# ---------------------------------------------------------------------------
# Catalog plotting styles
# ---------------------------------------------------------------------------
# Each entry defines a reusable visual identity: marker shape, color, size and
# transparency.  These are intentionally distinct so that overlapping sources
# from different catalogs remain readable on a sky map.

CATALOG_STYLES: Dict[str, Dict[str, Any]] = {
    "fermi": {
        "color": "limegreen",
        "marker": "^",
        "s": 80,
        "alpha": 0.9,
        "edgecolors": "darkgreen",
        "linewidths": 0.8,
        "label": "Fermi-LAT 4FGL",
    },
    "lhaaso_km2a": {
        "color": "crimson",
        "marker": "o",
        "s": 100,
        "alpha": 0.85,
        "edgecolors": "darkred",
        "linewidths": 1.0,
        "label": "LHAASO KM2A",
    },
    "lhaaso_wcda": {
        "color": "gold",
        "marker": "o",
        "s": 80,
        "alpha": 0.85,
        "edgecolors": "darkgoldenrod",
        "linewidths": 1.0,
        "label": "LHAASO WCDA",
    },
    "hawc": {
        "color": "royalblue",
        "marker": "s",
        "s": 90,
        "alpha": 0.85,
        "edgecolors": "navy",
        "linewidths": 1.0,
        "label": "HAWC 3HWC",
    },
    "hess": {
        "color": "darkorange",
        "marker": "D",
        "s": 80,
        "alpha": 0.85,
        "edgecolors": "saddlebrown",
        "linewidths": 1.0,
        "label": "HESS HGPS",
    },
}


def separate_lhaaso_catalog() -> Tuple[Table, Table]:
    km2a_catalog = Table(
        names=("name", "ra", "dec", "r39", "TS"), dtype=("str", "float64", "float64", "float64", "float64")
    )
    wcda_catalog = Table(
        names=("name", "ra", "dec", "r39", "TS"), dtype=("str", "float64", "float64", "float64", "float64")
    )
    lhaaso_catalog = SourceCatalog1LHAASO()
    source: SourceCatalogObject1LHAASO
    for source in lhaaso_catalog:
        data: dict[str, Any] = source.data
        if data["Model_a"].strip() == "KM2A":
            km2a_catalog.add_row(
                (f"{data['Source_Name']}_km2a", data["RAJ2000"], data["DECJ2000"], data["r39"], data["TS"])
            )
        elif data["Model_a"].strip() == "WCDA":
            wcda_catalog.add_row(
                (f"{data['Source_Name']}_wcda", data["RAJ2000"], data["DECJ2000"], data["r39"], data["TS"])
            )
        if data["Model_b"].strip() == "KM2A":
            km2a_catalog.add_row(
                (f"{data['Source_Name']}_km2a", data["RAJ2000_b"], data["DECJ2000_b"], data["r39_b"], data["TS_b"])
            )
        elif data["Model_b"].strip() == "WCDA":
            wcda_catalog.add_row(
                (f"{data['Source_Name']}_wcda", data["RAJ2000_b"], data["DECJ2000_b"], data["r39_b"], data["TS_b"])
            )
    return km2a_catalog, wcda_catalog


km2a_catalog, wcda_catalog = separate_lhaaso_catalog()


def _filter_table_by_region(table: Table, center: SkyCoord, radius: u.Quantity) -> Table:
    """Return rows of ``table`` within ``radius`` of ``center``.

    ``table`` must contain ``ra`` and ``dec`` columns in degrees.
    """
    coords = SkyCoord(ra=table["ra"], dec=table["dec"], unit="deg")
    mask = coords.separation(center) <= radius
    return table[mask]


def plot_lhaaso_catalog(ax: plt.Axes, center: SkyCoord, radius: u.Quantity):
    """Plot LHAASO 1LHAASO sources within ``radius`` of ``center``.

    KM2A and WCDA detections are shown with their own colors and markers.
    """
    km2a_in_region = _filter_table_by_region(km2a_catalog, center, radius)
    wcda_in_region = _filter_table_by_region(wcda_catalog, center, radius)

    if len(km2a_in_region) > 0:
        ax.scatter(
            km2a_in_region["ra"],
            km2a_in_region["dec"],
            **CATALOG_STYLES["lhaaso_km2a"],
        )
    if len(wcda_in_region) > 0:
        ax.scatter(
            wcda_in_region["ra"],
            wcda_in_region["dec"],
            **CATALOG_STYLES["lhaaso_wcda"],
        )


def plot_fermi_catalog(ax: plt.Axes, center: SkyCoord, radius: u.Quantity):
    """Plot Fermi-LAT 4FGL sources within ``radius`` of ``center``."""
    fermi_catalog = SourceCatalog4FGL()
    coords = SkyCoord(ra=fermi_catalog.table["RAJ2000"], dec=fermi_catalog.table["DEJ2000"], unit="deg")
    mask = coords.separation(center) <= radius
    table = fermi_catalog.table[mask]

    if len(table) > 0:
        ax.scatter(table["RAJ2000"], table["DEJ2000"], **CATALOG_STYLES["fermi"])


def plot_hawc_catalog(ax: plt.Axes, center: SkyCoord, radius: u.Quantity):
    """Plot HAWC 3HWC sources within ``radius`` of ``center``."""
    hawc_catalog = SourceCatalog3HWC()
    coords = SkyCoord(ra=hawc_catalog.table["ra"], dec=hawc_catalog.table["dec"], unit="deg")
    mask = coords.separation(center) <= radius
    table = hawc_catalog.table[mask]

    if len(table) > 0:
        ax.scatter(table["ra"], table["dec"], **CATALOG_STYLES["hawc"])


def plot_hess_catalog(ax: plt.Axes, center: SkyCoord, radius: u.Quantity):
    """Plot HESS HGPS sources within ``radius`` of ``center``."""
    hess_catalog = SourceCatalogHGPS()
    coords = SkyCoord(ra=hess_catalog.table["RAJ2000"], dec=hess_catalog.table["DEJ2000"], unit="deg")
    mask = coords.separation(center) <= radius
    table = hess_catalog.table[mask]

    if len(table) > 0:
        ax.scatter(table["RAJ2000"], table["DEJ2000"], **CATALOG_STYLES["hess"])
