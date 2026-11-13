from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from lst_tools.finder import MCDL2Metadata


@dataclass(frozen=True)
class GammaMCSample:
    """One gamma MC DL2 file and its simulation metadata."""

    path: Path
    mc_type: Literal["diffuse", "point-source"]
    nsb_level: float
    dec_line: int
    zenith: float
    azimuth: float

    @classmethod
    def from_metadata(cls, metadata: MCDL2Metadata) -> Self:
        return cls(
            path=metadata.path,
            mc_type="point-source" if metadata.mc_type == "Gamma" else "diffuse",
            nsb_level=metadata.nsb,
            dec_line=metadata.dec,
            zenith=metadata.theta,
            azimuth=metadata.azimuth,
        )

    @property
    def declination(self) -> float:
        return self.dec_line / 100

    @property
    def pointing_name(self) -> str:
        return f"theta_{self.zenith:g}_az_{self.azimuth:g}"


def select_gamma_mc_samples(
    metadata_records: Iterable[MCDL2Metadata],
    *,
    nsb_level: float,
    dec_line: int,
    irf_type: Literal["full-enclosure", "point"],
) -> tuple[GammaMCSample, ...]:
    """Select unique gamma MC samples matching one NSB tuning, declination line, and IRF type."""
    finder_mc_type = "Gamma" if irf_type == "point" else "GammaDiffuse"
    samples = (
        GammaMCSample.from_metadata(metadata)
        for metadata in metadata_records
        if metadata.nsb == nsb_level and metadata.dec == dec_line and metadata.mc_type == finder_mc_type
    )
    return tuple(dict.fromkeys(samples))
