import re
from pathlib import Path

from lst_tools.dl2 import LSTDL2EventTable

from .irf_nodes import IRFNode

MC_DL2_PATH = Path("/fefs/aswg/data/mc/DL2/AllSky")

_NSB_PATTERN = re.compile(r"(?:^|_)nsb_(?P<nsb>\d+(?:\.\d+)?)$")
_NODE_PATTERN = re.compile(r"node_theta_(?P<zenith>[+-]?\d+(?:\.\d+)?)_az_(?P<azimuth>[+-]?\d+(?:\.\d+)?)_*$")


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


def find_dl2_mc_path(
    base_path: str | Path,
    dl2_table: LSTDL2EventTable,
    *,
    diffuse: bool = True,
) -> list[IRFNode]:
    """Find the DL2 MC files matching an observation's NSB and declination.

    ``base_path`` may point to the AllSky DL2 root, a campaign directory, or
    its declination directory. By default, files are read from
    ``TestingDataset/GammaDiffuse``; pass ``diffuse=False`` to use
    ``TestingDataset/Gamma``. One node is returned for every merged DL2 file
    in a directory named ``node_theta_<zenith>_az_<azimuth>_``.
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

    declination = dec_line / 100
    intensity_cuts = dl2_table.intensity_cuts
    nodes = []
    for dec_path in _find_declination_directories(base_path, nsb_level, dec_line, diffuse):
        for node_path in sorted(dec_path.glob("node_theta_*_az_*")):
            if not node_path.is_dir() or (match := _NODE_PATTERN.fullmatch(node_path.name)) is None:
                continue

            zenith = float(match.group("zenith"))
            azimuth = float(match.group("azimuth"))
            for dl2_path in sorted(node_path.glob("dl2_*merged.h5")):
                if dl2_path.is_file():
                    nodes.append(
                        IRFNode(
                            declination=declination,
                            zenith=zenith,
                            azimuth=azimuth,
                            intensity_cuts=intensity_cuts,
                            dl2_path=dl2_path,
                        )
                    )

    return nodes
