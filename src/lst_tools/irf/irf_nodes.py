from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IRFNode:
    """One MC DL2 test node, i.e. one IRF file to generate.

    The node is identified only by where its input MC file lives:
    the NSB-tuning campaign, the declination line, and the pointing
    (zenith, azimuth) of the MC node. Analysis cuts such as the intensity
    cut or the gamma/hadron efficiency are *not* part of this identity;
    they are generation parameters supplied when the IRF is built.
    """

    nsb_level: float
    dec_line: int
    zenith: float
    azimuth: float
    dl2_path: Path

    @property
    def declination(self) -> float:
        return self.dec_line / 100

    @property
    def dec_name(self) -> str:
        if self.dec_line < 0:
            return f"dec_min_{abs(self.dec_line)}"
        return f"dec_{self.dec_line}"

    @property
    def pointing_name(self) -> str:
        return f"theta_{self.zenith:g}_az_{self.azimuth:g}"

    @property
    def file_name(self) -> str:
        return f"{self.pointing_name}.irf.fits.gz"

    @property
    def relative_path(self) -> Path:
        return Path(f"nsb_{self.nsb_level:g}") / self.dec_name / self.file_name
