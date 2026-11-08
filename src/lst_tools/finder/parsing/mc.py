"""Parse Monte Carlo DL2 paths in the standard AllSky layout."""

import re
from pathlib import Path
from typing import Literal

from ..metadata import MCDL2Attribute

_NSB_PATTERN = re.compile(r"(?:^|_)nsb_(?P<nsb>\d+(?:\.\d+)?)$")
_DEC_PATTERN = re.compile(r"dec_(?:(?P<negative>min)_)?(?P<digits>\d+)$")
_NODE_PATTERN = re.compile(
    r"node_(?:corsika_)?theta_(?P<theta>[+-]?\d+(?:\.\d+)?)_az_(?P<azimuth>[+-]?\d+(?:\.\d+)?)_*$"
)


def _testing_dataset_directories(path: str | Path) -> tuple[str, str, str]:
    parts = Path(path).parent.parts
    try:
        index = parts.index("TestingDataset")
        return parts[index + 1], parts[index + 2], parts[index + 3]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Expected TestingDataset/<type>/<dec>/<node> in: {path}") from error


def parse_mc_nsb(path: str | Path) -> float:
    for directory in Path(path).parent.parts:
        if match := _NSB_PATTERN.search(directory):
            return float(match.group("nsb"))
    raise ValueError(f"Could not parse MC NSB level from: {path}")


def parse_mc_type(path: str | Path) -> Literal["Gamma", "GammaDiffuse"]:
    mc_type = _testing_dataset_directories(path)[0]
    if mc_type == "Gamma":
        return "Gamma"
    if mc_type == "GammaDiffuse":
        return "GammaDiffuse"
    raise ValueError(f"Could not parse MC type from: {path}")


def parse_mc_dec(path: str | Path) -> int:
    dec_name = _testing_dataset_directories(path)[1]
    match = _DEC_PATTERN.fullmatch(dec_name)
    if match is None:
        raise ValueError(f"Could not parse MC declination from: {path}")
    dec = int(match.group("digits"))
    return -dec if match.group("negative") else dec


def parse_mc_pointing(path: str | Path) -> tuple[float, float]:
    node_name = _testing_dataset_directories(path)[2]
    match = _NODE_PATTERN.fullmatch(node_name)
    if match is None:
        raise ValueError(f"Could not parse MC pointing from: {path}")
    return float(match.group("theta")), float(match.group("azimuth"))


def parse_mc_dl2_path(path: str | Path) -> MCDL2Attribute:
    path = Path(path)
    theta, azimuth = parse_mc_pointing(path)
    return MCDL2Attribute(
        path=path,
        mc_type=parse_mc_type(path),
        theta=theta,
        azimuth=azimuth,
        nsb=parse_mc_nsb(path),
        dec=parse_mc_dec(path),
    )
