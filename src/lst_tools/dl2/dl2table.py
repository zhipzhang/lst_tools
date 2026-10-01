import re
from functools import cached_property
from os import PathLike
from pathlib import Path

import astropy.units as u
from lstchain.reco.utils import get_effective_time, get_intensity_cut

from ..pointing import Pointing
from .dl2table_base import LSTDL2TableBase


class LSTDL2EventTable(LSTDL2TableBase):
    """DL2 events and run metadata loaded from one lstchain file."""

    def __init__(self, file_name: str | PathLike[str]):
        self.run_id = self._parse_run_id(Path(file_name).name)
        super().__init__(file_name)
        self.t_eff, self.t_elapsed = get_effective_time(self.dl2_params)
        self.pointing = Pointing(
            self.dl2_params["alt_tel"].to_numpy(dtype=float),
            self.dl2_params["az_tel"].to_numpy(dtype=float),
            self.dl2_params["trigger_time"].to_numpy(dtype=float),
        )

    def __repr__(self) -> str:
        total_events = len(self.dl2_params)
        event_rates = total_events / self.t_eff
        events_above_100 = (self.dl2_params["intensity"] > 100).sum()
        rate_above_100 = events_above_100 / self.t_eff

        return (
            f"LSTDL2EventTable(run_id={self.run_id}, total_events={total_events}, "
            f"events_above_100={events_above_100}, event_rates={event_rates:.2f}, "
            f"rate_above_100={rate_above_100:.2f})"
        )

    @staticmethod
    def _parse_run_id(file_name: str) -> int:
        match = re.search(r"Run(?P<run_id>\d+)", file_name)
        if match is None:
            raise ValueError(f"could not find a run number in DL2 filename: {file_name}")
        return int(match.group("run_id"))

    @cached_property
    def pointing_ra(self) -> u.Quantity:
        return self.pointing.pointing_ra

    @cached_property
    def pointing_dec(self) -> u.Quantity:
        return self.pointing.pointing_dec

    @property
    def tailcut_level(self) -> tuple[int, int] | None:
        if self.dl1_file_path is None:
            return None
        match = re.search(r"tailcut(\d+)", self.dl1_file_path)
        if not match:
            raise ValueError(f"Could not parse tailcut level from input path: {self.dl1_file_path}")
        digits = match.group(1)
        mid = len(digits) // 2
        return int(digits[:mid]), int(digits[mid:])

    @property
    def intensity_cuts(self) -> float:
        intensity_cuts = get_intensity_cut(self.data)
        return (intensity_cuts // 10 + 1) * 10
