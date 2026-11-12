from .datacheck import DataCheckTables
from .datafilter import DataFilter

__all__ = [
    "DataCheckTables",
    "DataFilter",
    "plot_advanced_distributions",
]


def __getattr__(name: str):
    """Load plotting support only when it is requested."""
    if name == "plot_advanced_distributions":
        from .plot import plot_advanced_distributions

        globals()[name] = plot_advanced_distributions
        return plot_advanced_distributions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
