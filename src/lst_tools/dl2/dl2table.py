import re
from os import PathLike
from pathlib import Path

import astropy.units as u
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance
from lstchain.reco.utils import get_effective_time, get_intensity_cut

from ..pointing import Pointing


class LSTDL2EventTable:
    """DL2 events and run metadata loaded from one lstchain file."""

    def __init__(self, file_name: str | PathLike[str]):
        self.file_name = str(file_name)
        self.run_id = self._parse_run_id(Path(file_name).name)
        self.dl2_params: pd.DataFrame = pd.read_hdf(self.file_name, key=dl2_params_lstcam_key)
        self.t_eff, self.t_elapsed = get_effective_time(self.dl2_params)
        self.dl2_provenance = read_dl2_provenance(self.file_name)
        self.dl1_file_path: str | None = None
        self.rf_model_directory: str | None = None
        self.analyze_provenance()
        self.pointing = Pointing(
            self.dl2_params["alt_tel"],
            self.dl2_params["az_tel"],
            self.dl2_params["trigger_time"],
        )

    def __repr__(self) -> str:
        total_events = len(self.dl2_params)
        event_rates = total_events / self.t_eff
        return f"LSTDL2EventTable(run_id={self.run_id}, total_events={total_events}, event_rates={event_rates})"

    @staticmethod
    def _parse_run_id(file_name: str) -> int:
        match = re.search(r"Run(?P<run_id>\d+)", file_name)
        if match is None:
            raise ValueError(f"could not find a run number in DL2 filename: {file_name}")
        return int(match.group("run_id"))

    @property
    def data(self):
        return self.dl2_params

    @property
    def pointing_ra(self) -> u.Quantity:
        return self.pointing.pointing_ra

    @property
    def pointing_dec(self) -> u.Quantity:
        return self.pointing.pointing_dec

    def analyze_provenance(self):
        config = self.dl2_provenance.get("input", [])
        for input_info in config:
            roles = input_info.get("role", [])
            if not roles:
                continue
            if "input" in roles:
                self.dl1_file_path = input_info.get("url")
            if "model" in roles:
                self.rf_model_directory = input_info.get("url")

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
    def nsb_level(self) -> float | None:
        if self.rf_model_directory is None:
            return None
        match = re.search(r"nsb_tuning_([\d.]+)", self.rf_model_directory)
        return float(match.group(1)) if match else None

    @property
    def intensity_cuts(self) -> float:
        intensity_cuts = get_intensity_cut(self.data)
        return (intensity_cuts // 10 + 1) * 10

    @property
    def dec_line(self) -> int | None:
        if self.rf_model_directory is None:
            return None
        match = re.search(r"(?:^|/)dec_(?:(?P<negative>min)_)?(?P<digits>\d+)(?:/|$)", self.rf_model_directory)
        if match is None:
            return None
        dec_line = int(match.group("digits"))
        return -dec_line if match.group("negative") else dec_line
