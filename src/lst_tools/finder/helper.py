"""
Helper function for regular expression to parse the path for different levels.
"""

import re
from pathlib import Path
from typing import Literal

from lstchain.image.cleaning import find_tailcuts

from .datalevel_attributes import DL1Attributes, DL2Attributes, DL3Attributes, NSBLevel


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

    # Since all candidates are >= nsb_level, the closest higher one is simply the minimum candidate
    return min(candidates)


def parse_dl1_path(path: str, parse_nsb_level: bool = False) -> DL1Attributes:
    run_number = parse_run_id(path)
    tailcuts_level = parse_tailcuts_level(path)
    lstchain_version = parse_lstchain_version(path)
    # Call lstchain function to get nsb_level
    dl1_path = Path(path)
    if parse_nsb_level:
        directory = dl1_path.parent
        _, nsb_level, _ = find_tailcuts(directory, run_number)
        return DL1Attributes(dl1_path, run_number, tailcuts_level, lstchain_version, find_closest_nsb_level(nsb_level))
    else:
        nsb_level = None
        return DL1Attributes(dl1_path, run_number, tailcuts_level, lstchain_version, nsb_level)


def parse_dl2_path(path: str) -> DL2Attributes:
    run_number = parse_run_id(path)
    tailcuts_level = parse_tailcuts_level(path)
    lstchain_version = parse_lstchain_version(path)
    nsb_level = parse_nsb_level_from_path(path)
    return DL2Attributes(Path(path), run_number, tailcuts_level, lstchain_version, nsb_level)


def parse_dl3_path(path: str) -> DL3Attributes:
    return DL3Attributes(
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
