from .datacheck import DataCheckTables
from .datafilter import DataFilter
from .utils import assign_zenith_bins, validate_zenith_bin_edges, zenith_bin_labels

__all__ = [
    "DataCheckTables",
    "DataFilter",
    "assign_zenith_bins",
    "plot_advanced_distributions",
    "validate_zenith_bin_edges",
    "zenith_bin_labels",
]


def __getattr__(name: str):
    """Load plotting support only when it is requested."""
    if name == "plot_advanced_distributions":
        from .plot import plot_advanced_distributions

        globals()[name] = plot_advanced_distributions
        return plot_advanced_distributions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
