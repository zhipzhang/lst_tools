from pathlib import Path
from typing import Literal

from .samples import GammaMCSample


def gammaness_label(gh_type: Literal["efficiency", "cut"], gh_value: float) -> str:
    """Path-safe name of a gammaness selection."""
    prefix = "gheff" if gh_type == "efficiency" else "ghcut"
    return f"{prefix}{gh_value:g}"


def dec_name(dec_line: int) -> str:
    """Directory name of an MC declination line: ``dec_347``, ``dec_min_2276``."""
    if dec_line < 0:
        return f"dec_min_{abs(dec_line)}"
    return f"dec_{dec_line}"


def irf_grid_path(
    *,
    nsb_level: float,
    dec_line: int,
    irf_type: Literal["full-enclosure", "point"],
    cut_label: str,
) -> Path:
    """Directory of one IRF interpolation grid, relative to the IRF root."""
    return dec_name(dec_line) / Path(f"nsb_{nsb_level:g}") / irf_type / cut_label


def irf_file_path(
    sample: GammaMCSample,
    *,
    irf_type: Literal["full-enclosure", "point"],
    cut_label: str,
) -> Path:
    """Location of the IRF generated from ``sample`` under one configuration."""
    grid = irf_grid_path(
        nsb_level=sample.nsb_level,
        dec_line=sample.dec_line,
        irf_type=irf_type,
        cut_label=cut_label,
    )
    return grid / f"{sample.pointing_name}.irf.fits.gz"
