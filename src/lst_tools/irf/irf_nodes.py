from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from lst_tools.event_filter import GHType, gammaness_label


def dec_name(dec_line: int) -> str:
    """Directory name of an MC declination line: ``dec_347``, ``dec_min_2276``."""
    if dec_line < 0:
        return f"dec_min_{abs(dec_line)}"
    return f"dec_{dec_line}"


def irf_group_path(
    *,
    nsb_level: float,
    dec_line: int,
    irf_type: Literal["full-enclosure", "point"],
    cut_label: str,
) -> Path:
    """Directory of one IRF interpolation set, relative to the IRF root.

    Every pointing file in it shares NSB tuning, declination line, IRF type
    and gammaness selection, so the directory is exactly one interpolation
    set. IRF generation and DL3 reduction both get their path from here.
    """
    return Path(f"nsb_{nsb_level:g}") / dec_name(dec_line) / irf_type / cut_label


@dataclass(frozen=True)
class MCNode:
    """One MC DL2 test node: an input simulation, before any analysis cut.

    The node is identified by where its file lives and the IRF type it can
    produce: the NSB-tuning campaign, declination line, pointing (zenith,
    azimuth), and point-like or full-enclosure kind (which selects the
    ``Gamma`` or ``GammaDiffuse`` MC files). Analysis cuts are not part of
    the node — several IRFs, differing only in cuts, can be generated from
    the same MC node (see :class:`IRFNode`).

    The declination line is stored in hundredths of a degree, as in the MC
    campaign directory names (``dec_347`` is 3.47 deg).
    """

    nsb_level: float
    dec_line: int
    zenith: float
    azimuth: float
    dl2_path: Path
    irf_type: Literal["full-enclosure", "point"] = "full-enclosure"

    @property
    def declination(self) -> float:
        return self.dec_line / 100

    @property
    def dec_name(self) -> str:
        return dec_name(self.dec_line)

    @property
    def pointing_name(self) -> str:
        return f"theta_{self.zenith:g}_az_{self.azimuth:g}"


@dataclass(frozen=True)
class IRFNode(MCNode):
    """One IRF file to generate: an MC node plus the gammaness selection.

    The gammaness mode is part of the node's identity because it changes
    the IRF content: ``gh_type = "efficiency"`` generates energy-dependent
    cuts at ``gh_value``, ``gh_type = "cut"`` a fixed threshold. The label
    of that selection is a directory level, so IRFs of one MC node that
    differ only in gammaness cuts coexist instead of overwriting each
    other.
    """

    gh_type: GHType = "efficiency"
    gh_value: float = 0.7

    @property
    def cut_label(self) -> str:
        """Directory-safe name of the gammaness selection, e.g. ``gheff0.7``."""
        return gammaness_label(self.gh_type, self.gh_value)

    @property
    def file_name(self) -> str:
        return f"{self.pointing_name}.irf.fits.gz"

    @property
    def relative_path(self) -> Path:
        return (
            irf_group_path(
                nsb_level=self.nsb_level,
                dec_line=self.dec_line,
                irf_type=self.irf_type,
                cut_label=self.cut_label,
            )
            / self.file_name
        )

    @classmethod
    def from_mc_node(cls, node: MCNode, *, gh_type: GHType = "efficiency", gh_value: float = 0.7) -> "IRFNode":
        """The IRF of an MC node under one gammaness selection."""
        return cls(
            **{field.name: getattr(node, field.name) for field in fields(MCNode)},
            gh_type=gh_type,
            gh_value=gh_value,
        )
