import re
from pathlib import Path

from lst_tools.dl2 import LSTDL2EventTable

from .irf_nodes import IRFNode

MC_DL2_PATH = Path("/fefs/aswg/data/mc/DL2/AllSky")

_NSB_PATTERN = re.compile(r"(?:^|_)nsb_(?P<nsb>\d+(?:\.\d+)?)$")
_DEC_PATTERN = re.compile(r"dec_(?:(?P<negative>min)_)?(?P<digits>\d+)$")
_NODE_PATTERN = re.compile(
    r"node_(?:corsika_)?theta_(?P<zenith>[+-]?\d+(?:\.\d+)?)_az_(?P<azimuth>[+-]?\d+(?:\.\d+)?)_*$"
)


def irf_node_from_path(dl2_path: str | Path) -> IRFNode:
    """Parse the IRF node an MC DL2 file belongs to from its path.

    Expects the standard AllSky layout, with the NSB campaign, declination
    line, and pointing encoded in the directory names::

        .../nsb_<nsb>/.../dec_<line>/node_theta_<zenith>_az_<azimuth>_*/dl2_*.h5

    (``dec_min_<line>`` for negative declinations, ``node_corsika_...``
    for corsika nodes.)
    """
    dl2_path = Path(dl2_path)
    nsb_level = dec_line = zenith = azimuth = None
    for part in dl2_path.parts:
        if match := _NSB_PATTERN.search(part):
            nsb_level = float(match.group("nsb"))
        if match := _DEC_PATTERN.fullmatch(part):
            dec_line = -int(match.group("digits")) if match.group("negative") else int(match.group("digits"))
        if match := _NODE_PATTERN.fullmatch(part):
            zenith = float(match.group("zenith"))
            azimuth = float(match.group("azimuth"))

    if None in (nsb_level, dec_line, zenith, azimuth):
        raise ValueError(f"Could not parse an IRF node from the MC DL2 path: {dl2_path}")

    return IRFNode(
        nsb_level=nsb_level,
        dec_line=dec_line,
        zenith=zenith,
        azimuth=azimuth,
        dl2_path=dl2_path,
    )


def _campaign_matches_nsb(path: Path, nsb_level: float) -> bool:
    match = _NSB_PATTERN.search(path.name)
    return match is not None and float(match.group("nsb")) == nsb_level


def _find_declination_directories(
    base_path: Path,
    nsb_level: float,
    dec_line: int,
    diffuse: bool,
) -> list[Path]:
    dec_name = f"dec_min_{abs(dec_line)}" if dec_line < 0 else f"dec_{dec_line}"
    if base_path.name == dec_name:
        return [base_path]

    gamma_directory = "GammaDiffuse" if diffuse else "Gamma"
    relative_dec_path = Path("TestingDataset") / gamma_directory / dec_name
    if _campaign_matches_nsb(base_path, nsb_level):
        dec_path = base_path / relative_dec_path
        return [dec_path] if dec_path.is_dir() else []

    return sorted(
        dec_path
        for campaign_path in base_path.iterdir()
        if campaign_path.is_dir() and _campaign_matches_nsb(campaign_path, nsb_level)
        if (dec_path := campaign_path / relative_dec_path).is_dir()
    )


def find_irf_nodes(
    dl2_table: LSTDL2EventTable,
    base_path: str | Path = MC_DL2_PATH,
    *,
    diffuse: bool = True,
) -> list[IRFNode]:
    """Find the IRF nodes matching an observation's NSB level and declination line.

    The observation's DL2 table carries its NSB tuning and declination line
    through the RF model directory recorded in its provenance; both are used
    to locate the corresponding MC test dataset under ``base_path`` (the
    AllSky MC DL2 root, a campaign directory, or a declination directory).
    By default, files are read from ``TestingDataset/GammaDiffuse``; pass
    ``diffuse=False`` to use ``TestingDataset/Gamma``. One node is returned
    for every merged DL2 file in a directory named
    ``node_theta_<zenith>_az_<azimuth>_`` or
    ``node_corsika_theta_<zenith>_az_<azimuth>_``.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not base_path.is_dir():
        raise NotADirectoryError(base_path)

    nsb_level = dl2_table.nsb_level
    dec_line = dl2_table.dec_line
    if nsb_level is None:
        raise ValueError("DL2 table does not contain an NSB level")
    if dec_line is None:
        raise ValueError("DL2 table does not contain a declination line")

    nodes = []
    for dec_path in _find_declination_directories(base_path, nsb_level, dec_line, diffuse):
        for node_path in sorted(dec_path.glob("node_*theta_*_az_*")):
            if not node_path.is_dir() or _NODE_PATTERN.fullmatch(node_path.name) is None:
                continue

            for dl2_path in sorted(node_path.glob("dl2_*merged.h5")):
                if dl2_path.is_file():
                    nodes.append(irf_node_from_path(dl2_path))

    return nodes
