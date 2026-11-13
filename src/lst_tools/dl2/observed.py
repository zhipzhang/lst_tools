import math
import re
from functools import cached_property
from os import PathLike

import astropy.units as u
from lstchain.reco.utils import get_effective_time, get_intensity_cut

from lst_tools.finder import parse_run_id

from ..pointing import Pointing
from .base import LSTDL2Table


class LSTObservedDL2Table(LSTDL2Table):
    """DL2 events and run metadata loaded from one lstchain file."""

    def __init__(self, file_name: str | PathLike[str]):
        self.run_number = parse_run_id(str(file_name))
        super().__init__(file_name)

    @cached_property
    def _timing(self):
        return get_effective_time(self.events)

    @property
    def effective_time(self):
        return self._timing[0]

    @property
    def elapsed_time(self):
        return self._timing[1]

    @cached_property
    def pointing(self) -> Pointing:
        return Pointing(
            self.events["alt_tel"].to_numpy(dtype=float),
            self.events["az_tel"].to_numpy(dtype=float),
            self.events["trigger_time"].to_numpy(dtype=float),
        )

    def __repr__(self) -> str:
        total_events = len(self.events)
        event_rates = total_events / self.effective_time
        events_above_100 = (self.events["intensity"] > 100).sum()
        rate_above_100 = events_above_100 / self.effective_time

        return (
            f"LSTObservedDL2Table(run_number={self.run_number}, total_events={total_events}, "
            f"events_above_100={events_above_100}, event_rates={event_rates:.2f}, "
            f"rate_above_100={rate_above_100:.2f})"
        )

    @cached_property
    def pointing_ra(self) -> u.Quantity:
        return self.pointing.pointing_ra

    @cached_property
    def pointing_dec(self) -> u.Quantity:
        return self.pointing.pointing_dec

    @property
    def tailcuts_level(self) -> tuple[int, int] | None:
        if self.dl1_file_path is None:
            return None
        match = re.search(r"tailcut(\d+)", self.dl1_file_path)
        if not match:
            raise ValueError(f"Could not parse tailcut level from input path: {self.dl1_file_path}")
        digits = match.group(1)
        mid = len(digits) // 2
        return int(digits[:mid]), int(digits[mid:])

    @property
    def intensity_cut(self) -> float:
        intensity_cut = get_intensity_cut(self.events)
        return math.ceil(intensity_cut / 10) * 10
