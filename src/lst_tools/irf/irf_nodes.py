from dataclasses import dataclass
from pathlib import Path


@dataclass
class IRFNode:
    declination: float
    zenith: float
    azimuth: float
    intensity_cuts: float
    dl2_path: Path | None = None
    gh_efficiency: float | None = None

    @property
    def dec_name(self):
        dec_line = round(self.declination * 100)
        if dec_line < 0:
            return f"dec_min_{abs(dec_line)}"
        return f"dec_{dec_line}"

    @property
    def pointing_name(self):
        return f"azimuth_{self.azimuth:.1f}_zenith_{self.zenith:.1f}"

    @property
    def path_name(self):
        path_name = (
            f"{self.dec_name}/intensity_{self.intensity_cuts:.0f}/gh_eff_{self.gh_efficiency:.1f}/{self.pointing_name}"
        )
        return path_name
