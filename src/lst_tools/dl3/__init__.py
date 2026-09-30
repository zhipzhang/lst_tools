"""Utilities for discovering and selecting LST DL3 data products."""

from .products import (
    DL3Product,
    DL3Request,
    discover_lst_dl3_products,
    load_dl3_requests,
    parse_dl3_path,
    select_configured_dl3_products,
)

__all__ = [
    "DL3Product",
    "DL3Request",
    "Dl3DataStore",
    "discover_lst_dl3_products",
    "load_dl3_requests",
    "parse_dl3_path",
    "select_configured_dl3_products",
]


def __getattr__(name: str):
    """Load the heavier Gammapy data-store support only when requested."""
    if name == "Dl3DataStore":
        from .data_store import Dl3DataStore

        globals()[name] = Dl3DataStore
        return Dl3DataStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
