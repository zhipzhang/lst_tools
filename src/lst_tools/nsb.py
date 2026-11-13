"""Night-sky-background levels used by LST data products."""

from enum import Enum


class NSBLevel(float, Enum):
    """Predefined NSB levels inheriting from float for easy comparison/math."""

    L0_00 = 0.0
    L0_07 = 0.07
    L0_14 = 0.14
    L0_22 = 0.22
    L0_38 = 0.38
    L0_50 = 0.50
    L0_81 = 0.81
    L1_25 = 1.25
    L1_76 = 1.76
    L2_34 = 2.34


def nsb_level_at_or_above(nsb_level: float, strict: bool = False) -> NSBLevel:
    """Return the lowest defined NSB level at or above the input value.

    If ``strict`` is true, the selected level must be strictly greater than
    the input value.
    """
    candidates = [level for level in NSBLevel if (level > nsb_level if strict else level >= nsb_level)]

    if not candidates:
        raise ValueError(f"No NSBLevel found higher than {nsb_level}. Maximum available is {max(NSBLevel).value}.")

    return min(candidates)


find_closest_nsb_level = nsb_level_at_or_above
