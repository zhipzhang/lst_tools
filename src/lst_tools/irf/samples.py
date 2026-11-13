from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from lst_tools.finder import MCDL2Attribute


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
    def from_attribute(cls, attribute: MCDL2Attribute) -> Self:
        return cls(
            path=attribute.path,
            mc_type="point-source" if attribute.mc_type == "Gamma" else "diffuse",
            nsb_level=attribute.nsb,
            dec_line=attribute.dec,
            zenith=attribute.theta,
            azimuth=attribute.azimuth,
        )

    @property
    def declination(self) -> float:
        return self.dec_line / 100

    @property
    def pointing_name(self) -> str:
        return f"theta_{self.zenith:g}_az_{self.azimuth:g}"


def select_gamma_mc_samples(
    attributes: Iterable[MCDL2Attribute],
    *,
    nsb_level: float,
    dec_line: int,
    irf_type: Literal["full-enclosure", "point"],
) -> tuple[GammaMCSample, ...]:
    """Select unique gamma MC samples matching one NSB tuning, declination line, and IRF type."""
    finder_mc_type = "Gamma" if irf_type == "point" else "GammaDiffuse"
    samples = (
        GammaMCSample.from_attribute(attribute)
        for attribute in attributes
        if attribute.nsb == nsb_level and attribute.dec == dec_line and attribute.mc_type == finder_mc_type
    )
    return tuple(dict.fromkeys(samples))
