"""Resolve NSB levels from DL1 run data."""

from os import PathLike

from lstchain.image.cleaning import find_tailcuts

from lst_tools.finder import parse_dl1_path
from lst_tools.nsb import NSBLevel, nsb_level_at_or_above


def resolve_dl1_nsb_level(path: str | PathLike[str]) -> NSBLevel:
    """Estimate the NSB level used for a DL1 run by reading its subrun data.

    This validates the DL1 path metadata and then reads the run's pedestal
    events to derive an NSB tuning level, so it is considerably more expensive
    than path parsing.
    """
    dl1_metadata = parse_dl1_path(str(path))
    _, nsb_level, _ = find_tailcuts(dl1_metadata.path.parent, dl1_metadata.run_number)
    return nsb_level_at_or_above(nsb_level)
