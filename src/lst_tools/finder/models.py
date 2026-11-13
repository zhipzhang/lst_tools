"""Metadata models associated with LST data files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from lst_tools.nsb import NSBLevel


@dataclass
class ObservedDataMetadata:
    path: Path
    run_number: int
    tailcuts_level: tuple[int, int]
    lstchain_version: str
    nsb_level: NSBLevel | None

    def __post_init__(self) -> None:
        if isinstance(self.nsb_level, float):
            self.nsb_level = NSBLevel(float(self.nsb_level))


DL1Metadata: TypeAlias = ObservedDataMetadata
DL2Metadata: TypeAlias = ObservedDataMetadata


@dataclass(frozen=True)
class MCDL2Metadata:
    path: Path
    mc_type: Literal["Gamma", "GammaDiffuse"]
    theta: float
    azimuth: float
    nsb: float
    dec: int


@dataclass
class DL3Metadata(ObservedDataMetadata):
    """Metadata for DL3 product variants on the La Palma server.

    Variants include full-enclosure/point-like IRFs and different
    gamma/hadron and theta efficiencies.
    """

    irf_type: Literal["full-enclosure", "point"]
    mc_data_type: Literal["ring-wobble", "diffuse"]
    gh_eff: float
    theta_containment: float
