"""Parse paths and filenames for observed LST data."""

import re
from pathlib import Path
from typing import Literal

from lst_tools.nsb import NSBLevel

from ..models import DL1Metadata, DL2Metadata, DL3Metadata


def parse_run_id(path: str) -> int:
    match = re.search(r"Run(\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse run number from: {path!r}")
    return int(match.group(1))


def parse_lstchain_version(path: str) -> str:
    match = re.search(r"v(\d+\.\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse lstchain version from: {path!r}")
    return match.group(1)


def parse_tailcuts_level(path: str) -> tuple[int, int]:
    match = re.search(r"tailcut(\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse tailcuts level from: {path!r}")
    digits = match.group(1)
    mid = len(digits) // 2
    return int(digits[:mid]), int(digits[mid:])


def parse_nsb_level_from_path(path: str) -> NSBLevel:
    match = re.search(r"nsb_tuning_(\d+\.\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse nsb level from: {path!r}")
    return NSBLevel(float(match.group(1)))


def parse_gh_eff(path: str) -> float:
    match = re.search(r"gheff(\d+\.\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse gh efficiency from: {path!r}")
    return float(match.group(1))


def parse_theta_containment(path: str) -> float:
    match = re.search(r"thetacont(\d+\.\d+)", path)
    if match is None:
        raise ValueError(f"Could not parse theta containment from: {path!r}")
    return float(match.group(1))


def parse_irf_type(path: str) -> Literal["full-enclosure", "point"]:
    parts = Path(path).parts
    if "full-enclosure" in parts:
        return "full-enclosure"
    if "point" in parts:
        return "point"
    raise ValueError(f"Could not parse irf type from: {path!r}")


def parse_mc_data_type(path: str) -> Literal["ring-wobble", "diffuse"]:
    parts = Path(path).parts
    if "ring-wobble" in parts:
        return "ring-wobble"
    if "diffuse" in parts:
        return "diffuse"
    raise ValueError(f"Could not parse mc data type from: {path!r}")


def parse_dl1_path(path: str) -> DL1Metadata:
    run_number = parse_run_id(path)
    tailcuts_level = parse_tailcuts_level(path)
    lstchain_version = parse_lstchain_version(path)
    return DL1Metadata(Path(path), run_number, tailcuts_level, lstchain_version, None)


def parse_dl2_path(path: str) -> DL2Metadata:
    run_number = parse_run_id(path)
    tailcuts_level = parse_tailcuts_level(path)
    lstchain_version = parse_lstchain_version(path)
    nsb_level = parse_nsb_level_from_path(path)
    return DL2Metadata(Path(path), run_number, tailcuts_level, lstchain_version, nsb_level)


def parse_dl3_path(path: str) -> DL3Metadata:
    return DL3Metadata(
        path=Path(path),
        run_number=parse_run_id(path),
        tailcuts_level=parse_tailcuts_level(path),
        lstchain_version=parse_lstchain_version(path),
        nsb_level=parse_nsb_level_from_path(path),
        irf_type=parse_irf_type(path),
        mc_data_type=parse_mc_data_type(path),
        gh_eff=parse_gh_eff(path),
        theta_containment=parse_theta_containment(path),
    )
