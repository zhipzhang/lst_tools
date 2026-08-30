from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MCNode:
    declination: float
    zenith: float
    azimuth: float
    dl2_path: Path

    @property
    def dec_name(self):
        # e.g. 22.76 -> dec_2276
        return f"dec_{round(self.declination * 100):04d}"

    @property
    def pointing_name(self):
        return f"azimuth_{self.azimuth:.1f}_zenith_{self.zenith:.1f}"
