from .source import CatalogSource
from .utils import select_runs_away_from_sources

__all__ = [
    "CATALOG_STYLES",
    "CatalogSource",
    "SkyPlotter",
    "load_fermi_sources",
    "load_hawc_sources",
    "load_hess_sources",
    "load_lhaaso_sources",
    "plot_fermi_catalog",
    "plot_hawc_catalog",
    "plot_hess_catalog",
    "plot_lhaaso_catalog",
    "plot_sources",
    "select_region",
    "select_runs_away_from_sources",
]

_PLOT_EXPORTS = {
    "CATALOG_STYLES",
    "SkyPlotter",
    "plot_fermi_catalog",
    "plot_hawc_catalog",
    "plot_hess_catalog",
    "plot_lhaaso_catalog",
    "plot_sources",
}
_LOADER_EXPORTS = {
    "load_fermi_sources",
    "load_hawc_sources",
    "load_hess_sources",
    "load_lhaaso_sources",
    "select_region",
}


def __getattr__(name: str):
    """Load optional catalog data and plotting support only when requested."""
    if name in _LOADER_EXPORTS:
        from . import loaders

        value = getattr(loaders, name)
        globals()[name] = value
        return value
    if name in _PLOT_EXPORTS:
        from . import plot

        value = getattr(plot, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
