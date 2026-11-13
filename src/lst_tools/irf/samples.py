from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class GammaMCSample:
    """One gamma MC DL2 file and its simulation metadata."""

    path: Path
    mc_type: Literal["diffuse", "point-source"]
    nsb_level: float
    dec_line: int
    zenith: float
    azimuth: float

    @property
    def declination(self) -> float:
        return self.dec_line / 100

    @property
    def pointing_name(self) -> str:
        return f"theta_{self.zenith:g}_az_{self.azimuth:g}"
