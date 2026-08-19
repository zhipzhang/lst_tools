import astropy.units as u
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from astropy.coordinates import SkyCoord

from lst_ulities.catalog import (
    CATALOG_STYLES,
    CatalogSource,
    SkyPlotter,
    plot_sources,
    select_region,
)

CRAB = SkyCoord(ra=83.6331, dec=22.0145, unit="deg")

SOURCES = (
    CatalogSource(name="point_like", coord=CRAB, catalog="hawc"),
    CatalogSource(name="extended", coord=CRAB, catalog="hess", extension=0.5 * u.deg),
    CatalogSource(name="far_away", coord=SkyCoord(ra=200, dec=0, unit="deg"), catalog="fermi"),
)


def test_catalog_source_is_extended():
    point_like, extended, _ = SOURCES
    assert not point_like.is_extended
    assert extended.is_extended
    zero_extension = CatalogSource(name="zero", coord=CRAB, catalog="hess", extension=0 * u.deg)
    assert not zero_extension.is_extended


def test_select_region():
    selected = select_region(SOURCES, center=CRAB, radius=1 * u.deg)
    assert [s.name for s in selected] == ["point_like", "extended"]
    selected = select_region(SOURCES, center=SOURCES[2].coord, radius=0.1 * u.deg)
    assert [s.name for s in selected] == ["far_away"]
    assert select_region((), center=CRAB, radius=1 * u.deg) == []


def test_catalog_styles_cover_known_catalogs():
    assert set(CATALOG_STYLES) == {"fermi", "lhaaso_km2a", "lhaaso_wcda", "hawc", "hess"}
    for style in CATALOG_STYLES.values():
        for key in ("color", "marker", "s", "alpha", "zorder", "label"):
            assert key in style


def test_plot_sources_on_plain_axes():
    fig, ax = plt.subplots()
    plot_sources(ax, SOURCES, center=CRAB, radius=1 * u.deg)
    # one scatter per catalog group present in the region (hawc + hess)
    assert len(ax.collections) == 2
    # one extension circle for the extended source
    assert len(ax.patches) == 1
    # both nearby sources are labeled
    assert {t.get_text() for t in ax.texts} == {"point_like", "extended"}
    plt.close(fig)


def test_skyplotter_without_labels_or_circles():
    fig, ax = plt.subplots()
    plotter = SkyPlotter(ax)
    plotter.draw_sources(list(SOURCES[:2]), label_sources=False, min_extension=1 * u.deg)
    assert len(ax.collections) == 2
    assert len(ax.patches) == 0
    assert len(ax.texts) == 0
    plt.close(fig)


def test_unknown_catalog_style_raises():
    fig, ax = plt.subplots()
    bad = CatalogSource(name="bad", coord=CRAB, catalog="unknown")
    with pytest.raises(KeyError):
        SkyPlotter(ax).draw_sources([bad])
    plt.close(fig)
