"""
Plotting utilities for gamma-ray source catalogs.

The module is organized in three layers:

1. **Data layer** – each catalog is loaded once (lazily and cached) into a
   uniform tuple of :class:`CatalogSource`, which carries the source name,
   the ICRS position and an optional angular extension (``None`` for
   point-like sources).  ``select_region`` filters any source tuple to a
   circular sky region.

2. **Transform layer** – :class:`SkyPlotter` wraps a matplotlib Axes and
   hides the differences between plain Axes and WCSAxes:

   - ``scatter`` accepts ``transform=ax.get_transform("icrs")``; passing
     ``None`` on plain Axes is fine,
   - text annotations must receive the transform as ``xycoords`` (the
     ``transform`` keyword of ``annotate`` silently drops text on WCSAxes),
   - ``Circle`` patches need an explicit ``set_transform``.

3. **Drawing layer** – point sources are scattered with the per-catalog
   marker styles defined in :data:`CATALOG_STYLES`, extended sources get an
   additional dashed circle with their angular radius, and every source can
   be labeled with its catalog name using white-halo text.

Supported catalogs: Fermi-LAT 4FGL, LHAASO 1LHAASO (KM2A/WCDA),
HAWC 3HWC, HESS HGPS.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.catalog import (
    SourceCatalog1LHAASO,
    SourceCatalog3HWC,
    SourceCatalog4FGL,
    SourceCatalogHGPS,
)
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Catalog plotting styles
# ---------------------------------------------------------------------------
# Each entry defines a reusable visual identity: marker shape, color, size and
# transparency.  These are intentionally distinct so that overlapping sources
# from different catalogs remain readable on a sky map.  Marker edge colors
# double as the color for labels and extension circles.

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


def _deg(value: Any) -> float:
    """Convert a plain number or a Quantity to decimal degrees."""
    return float(u.Quantity(value, u.deg).to_value(u.deg))


# ---------------------------------------------------------------------------
# Data layer: uniform catalog sources
# ---------------------------------------------------------------------------


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


def select_region(sources: Tuple[CatalogSource, ...], center: SkyCoord, radius: u.Quantity) -> List[CatalogSource]:
    """Return the sources within ``radius`` of ``center``."""
    if len(sources) == 0:
        return []
    coords = SkyCoord([s.coord.icrs for s in sources])
    separations = coords.separation(center)
    return [s for s, sep in zip(sources, separations) if sep <= radius]


# ---------------------------------------------------------------------------
# Transform + drawing layer
# ---------------------------------------------------------------------------


class SkyPlotter:
    """Draw catalog sources on a plain matplotlib Axes or a WCSAxes.

    All WCS-related special cases are handled here, so drawing code only
    deals with ICRS degrees.
    """

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._transform = self._detect_icrs_transform(ax)

    @staticmethod
    def _detect_icrs_transform(ax: plt.Axes):
        """Return the ICRS transform of a WCSAxes, or ``None`` for plain axes."""
        try:
            return ax.get_transform("icrs")
        except (AttributeError, TypeError, ValueError):
            return None

    def draw_sources(
        self,
        sources: List[CatalogSource],
        *,
        label_sources: bool = True,
        min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
    ):
        """Draw sources grouped by catalog so legend entries appear once."""
        groups: Dict[str, List[CatalogSource]] = {}
        for source in sources:
            groups.setdefault(source.catalog, []).append(source)

        for catalog_key, group in groups.items():
            style = CATALOG_STYLES[catalog_key]
            edge_color = style.get("edgecolors", style["color"])
            self._scatter(group, style)
            for source in group:
                if source.is_extended and source.extension >= min_extension:
                    self._draw_extension_circle(source, edge_color)
                if label_sources:
                    self._draw_label(source, edge_color)

    def _scatter(self, sources: List[CatalogSource], style: Dict[str, Any]):
        ra = [s.coord.ra.deg for s in sources]
        dec = [s.coord.dec.deg for s in sources]
        self.ax.scatter(ra, dec, transform=self._transform, **style)

    def _draw_label(self, source: CatalogSource, color: str):
        text = self.ax.annotate(
            source.name,
            (source.coord.ra.deg, source.coord.dec.deg),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="medium",
            color=color,
            xycoords=self._transform or "data",
            zorder=6,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])

    def _draw_extension_circle(self, source: CatalogSource, color: str):
        circle = Circle(
            (source.coord.ra.deg, source.coord.dec.deg),
            source.extension.to_value("deg"),
            fill=False,
            linestyle="--",
            linewidth=1.3,
            edgecolor=color,
            alpha=0.85,
            zorder=3,
        )
        if self._transform is not None:
            circle.set_transform(self._transform)
        self.ax.add_patch(circle)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_sources(
    ax: plt.Axes,
    sources: Tuple[CatalogSource, ...],
    center: SkyCoord,
    radius: u.Quantity,
    *,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot any tuple of :class:`CatalogSource` within ``radius`` of ``center``.

    Point sources are drawn as markers, extended sources (``extension >=
    min_extension``) additionally get a dashed circle, and every source is
    labeled by name unless ``label_sources`` is ``False``.
    """
    selected = select_region(sources, center, radius)
    SkyPlotter(ax).draw_sources(selected, label_sources=label_sources, min_extension=min_extension)


def plot_lhaaso_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot LHAASO 1LHAASO sources within ``radius`` of ``center``.

    KM2A and WCDA detections are shown with their own colors and markers;
    sources with ``r39 >= min_extension`` get a dashed 39% containment
    circle.  Works on both plain matplotlib axes and WCSAxes.
    """
    plot_sources(ax, load_lhaaso_sources(), center, radius, label_sources=label_sources, min_extension=min_extension)


def plot_fermi_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot Fermi-LAT 4FGL sources within ``radius`` of ``center``.

    Extended sources larger than ``min_extension`` get a dashed circle with
    the model semi-major radius.  Works on both plain axes and WCSAxes.
    """
    plot_sources(ax, load_fermi_sources(), center, radius, label_sources=label_sources, min_extension=min_extension)


def plot_hawc_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
):
    """Plot HAWC 3HWC sources within ``radius`` of ``center``.

    Works on both plain matplotlib axes and WCSAxes.
    """
    plot_sources(ax, load_hawc_sources(), center, radius, label_sources=label_sources)


def plot_hess_catalog(
    ax: plt.Axes,
    center: SkyCoord,
    radius: u.Quantity,
    label_sources: bool = True,
    min_extension: u.Quantity = DEFAULT_MIN_EXTENSION,
):
    """Plot HESS HGPS sources within ``radius`` of ``center``.

    Sources with a Gaussian ``Size >= min_extension`` get a dashed extension
    circle.  Works on both plain matplotlib axes and WCSAxes.
    """
    plot_sources(ax, load_hess_sources(), center, radius, label_sources=label_sources, min_extension=min_extension)
