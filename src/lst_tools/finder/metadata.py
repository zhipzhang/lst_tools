"""Metadata and domain values associated with LST data files."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias


class NSBLevel(float, Enum):
    """Predefined NSB levels inheriting from float for easy comparison/math."""

    L0_00 = 0.0
    L0_07 = 0.07
    L0_14 = 0.14
    L0_22 = 0.22
    L0_38 = 0.38
    L0_50 = 0.50
    L0_81 = 0.81
    L1_25 = 1.25
    L1_76 = 1.76
    L2_34 = 2.34


def find_closest_nsb_level(nsb_level: float, strict: bool = False) -> NSBLevel:
    """Find the closest NSBLevel that is higher than the provided value.

    Args:
        nsb_level: The input float value.
        strict: If True, looks for level > nsb_level.
                If False (default), looks for level >= nsb_level.

    Raises:
        ValueError: If nsb_level exceeds all defined enum levels.
    """
    candidates = [level for level in NSBLevel if (level > nsb_level if strict else level >= nsb_level)]

    if not candidates:
        raise ValueError(f"No NSBLevel found higher than {nsb_level}. Maximum available is {max(NSBLevel).value}.")

    return min(candidates)


@dataclass
class DLAttributes:
    path: Path
    run_number: int
    tailcuts_level: tuple[int, int]
    lstchain_version: str
    nsb_level: NSBLevel | None

    def __post_init__(self) -> None:
        if isinstance(self.nsb_level, float):
            self.nsb_level = NSBLevel(float(self.nsb_level))


DL1Attributes: TypeAlias = DLAttributes
DL2Attributes: TypeAlias = DLAttributes


@dataclass(frozen=True)
class MCDL2Attribute:
    path: Path
    mc_type: Literal["Gamma", "GammaDiffuse"]
    theta: float
    azimuth: float
    nsb: float
    dec: int


@dataclass
class DL3Attributes(DLAttributes):
    """Attributes for the DL3 product variants on the La Palma server.

    Variants include full-enclosure/point-like IRFs and different
    gamma/hadron and theta efficiencies.
    """

    irf_type: Literal["full-enclosure", "point"]
    mc_data_type: Literal["ring-wobble", "diffuse"]
    gh_eff: float
    theta_containment: float
