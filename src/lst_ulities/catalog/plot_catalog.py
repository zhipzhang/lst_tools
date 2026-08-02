"""
Helper functions for plotting catalog data.

Support catalog:
    - Fermi-LAT 4FGL
    - LHAASO 1LHAASO
    - HESS HESS-Catalog
"""

from typing import Any, Dict, Optional, Tuple

import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.catalog import (
    SourceCatalog1LHAASO,
    SourceCatalog3HWC,
    SourceCatalog4FGL,
    SourceCatalogHGPS,
    SourceCatalogObject1LHAASO,
)
from matplotlib.patches import Circle

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
        "s": 55,
        "alpha": 0.9,
        "edgecolors": "darkgreen",
        "linewidths": 0.8,
        "zorder": 4,
        "label": "Fermi-LAT 4FGL",
    },
    "lhaaso_km2a": {
        "color": "crimson",
        "marker": "o",
        "s": 70,
        "alpha": 0.9,
        "edgecolors": "darkred",
        "linewidths": 1.0,
        "zorder": 5,
        "label": "LHAASO KM2A",
    },
    "lhaaso_wcda": {
        "color": "gold",
        "marker": "o",
        "s": 55,
        "alpha": 0.9,
        "edgecolors": "darkgoldenrod",
        "linewidths": 1.0,
        "zorder": 5,
        "label": "LHAASO WCDA",
    },
    "hawc": {
        "color": "royalblue",
        "marker": "s",
        "s": 60,
        "alpha": 0.9,
        "edgecolors": "navy",
        "linewidths": 1.0,
        "zorder": 4,
        "label": "HAWC 3HWC",
    },
    "hess": {
        "color": "darkorange",
        "marker": "D",
        "s": 55,
        "alpha": 0.9,
        "edgecolors": "saddlebrown",
        "linewidths": 1.0,
        "zorder": 4,
        "label": "HESS HGPS",
    },
}

DEFAULT_MIN_EXTENSION = 0.1 * u.deg


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


def _get_icrs_transform(ax: plt.Axes):
    """Return the ICRS transform of a WCSAxes, or ``None`` for plain axes."""
    try:
        return ax.get_transform("icrs")
    except (AttributeError, TypeError, ValueError):
        return None


def _scatter_sky(ax: plt.Axes, ra: u.Quantity, dec: u.Quantity, **kwargs):
    """Scatter points that may be defined on a WCSAxes.

    ``ra`` and ``dec`` are interpreted as ICRS degrees.  If ``ax`` is a
    WCSAxes, the ICRS-to-pixel transform is applied automatically so the
    sources land on the correct image pixels.  For plain matplotlib axes the
    values are plotted directly.
    """
    ax.scatter(ra, dec, transform=_get_icrs_transform(ax), **kwargs)


def _add_source_label(ax: plt.Axes, ra_deg: float, dec_deg: float, name: str, color: str):
    """Annotate a source name next to its position with a white halo."""
    text = ax.annotate(
        name,
        (ra_deg, dec_deg),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.5,
        fontweight="medium",
        color=color,
        xycoords=_get_icrs_transform(ax) or "data",
        zorder=6,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])


def _draw_extension_circle(ax: plt.Axes, ra_deg: float, dec_deg: float, radius_deg: float, color: str):
    """Draw a dashed circle showing the spatial extension of a source."""
    circle = Circle(
        (ra_deg, dec_deg),
        radius_deg,
        fill=False,
        linestyle="--",
        linewidth=1.3,
        edgecolor=color,
        alpha=0.85,
        zorder=3,
    )
    transform = _get_icrs_transform(ax)
    if transform is not None:
        circle.set_transform(transform)
    ax.add_patch(circle)


def _plot_sources(
    ax: plt.Axes,
    table: Table,
    *,
    ra_col: str,
    dec_col: str,
    name_col: str,
    style: Dict[str, Any],
    ext_col: Optional[str] = None,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
    label_sources: bool = True,
):
    """Scatter catalog sources, label them and mark extended ones."""
    if len(table) == 0:
        return

    edge_color = style.get("edgecolors", style["color"])
    _scatter_sky(ax, table[ra_col], table[dec_col], **style)

    min_extension_deg = min_extension.to_value("deg")
    for row in table:
        ra_deg, dec_deg = float(row[ra_col]), float(row[dec_col])
        if ext_col is not None:
            ext = row[ext_col]
            if np.isfinite(ext) and ext >= min_extension_deg:
                _draw_extension_circle(ax, ra_deg, dec_deg, float(ext), edge_color)
        if label_sources:
            _add_source_label(ax, ra_deg, dec_deg, str(row[name_col]), edge_color)


def plot_lhaaso_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot LHAASO 1LHAASO sources within ``radius`` of ``center``.

    KM2A and WCDA detections are shown with their own colors and markers.
    Each source is labeled by name, and sources with ``r39`` larger than
    ``min_extension`` get a dashed circle showing their 39% containment
    radius.  Works on both plain matplotlib axes and WCSAxes.
    """
    for table, style_key in ((km2a_catalog, "lhaaso_km2a"), (wcda_catalog, "lhaaso_wcda")):
        _plot_sources(
            ax,
            _filter_table_by_region(table, center, radius),
            ra_col="ra",
            dec_col="dec",
            name_col="name",
            style=CATALOG_STYLES[style_key],
            ext_col="r39",
            min_extension=min_extension,
            label_sources=label_sources,
        )


def plot_fermi_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot Fermi-LAT 4FGL sources within ``radius`` of ``center``.

    Each source is labeled by name, and extended sources larger than
    ``min_extension`` get a dashed circle showing their semi-major radius.
    Works on both plain matplotlib axes and WCSAxes.
    """
    fermi_catalog = SourceCatalog4FGL()
    table = fermi_catalog.table
    coords = SkyCoord(ra=table["RAJ2000"], dec=table["DEJ2000"], unit="deg")
    table = table[coords.separation(center) <= radius]

    _plot_sources(
        ax,
        table,
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        name_col="Source_Name",
        style=CATALOG_STYLES["fermi"],
        label_sources=label_sources,
    )

    extended = fermi_catalog.extended_sources_table
    ext_coords = SkyCoord(ra=extended["RAJ2000"], dec=extended["DEJ2000"], unit="deg")
    extended = extended[ext_coords.separation(center) <= radius]
    min_extension_deg = min_extension.to_value("deg")
    edge_color = CATALOG_STYLES["fermi"]["edgecolors"]
    for row in extended:
        semi_major = row["Model_SemiMajor"]
        if np.isfinite(semi_major) and semi_major >= min_extension_deg:
            _draw_extension_circle(ax, float(row["RAJ2000"]), float(row["DEJ2000"]), float(semi_major), edge_color)


def plot_hawc_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
):
    """Plot HAWC 3HWC sources within ``radius`` of ``center``.

    Each source is labeled by name.  Works on both plain matplotlib axes
    and WCSAxes.
    """
    hawc_catalog = SourceCatalog3HWC()
    table = hawc_catalog.table
    coords = SkyCoord(ra=table["ra"], dec=table["dec"], unit="deg")
    table = table[coords.separation(center) <= radius]

    _plot_sources(
        ax,
        table,
        ra_col="ra",
        dec_col="dec",
        name_col="source_name",
        style=CATALOG_STYLES["hawc"],
        label_sources=label_sources,
    )


def plot_hess_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot HESS HGPS sources within ``radius`` of ``center``.

    Each source is labeled by name, and sources with a Gaussian ``Size``
    larger than ``min_extension`` get a dashed circle showing their
    extension.  Works on both plain matplotlib axes and WCSAxes.
    """
    hess_catalog = SourceCatalogHGPS()
    table = hess_catalog.table
    coords = SkyCoord(ra=table["RAJ2000"], dec=table["DEJ2000"], unit="deg")
    table = table[coords.separation(center) <= radius]

    _plot_sources(
        ax,
        table,
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        name_col="Source_Name",
        style=CATALOG_STYLES["hess"],
        ext_col="Size",
        min_extension=min_extension,
        label_sources=label_sources,
    )
