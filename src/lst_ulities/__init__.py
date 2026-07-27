"""lst_ulities: utilities built on top of lstchain and ctapipe."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lst_ulities")
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"
