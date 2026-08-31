import re
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance
from lstchain.reco.utils import get_effective_time, get_intensity_cut


class LSTDL2EventTable:
    """DL2 events and run metadata loaded from one lstchain file.

    Pointing altitude, azimuth, and zenith are floats in radians.
    """

    _required_pointing_columns = frozenset({"alt_tel", "az_tel"})

    def __init__(self, file_name: str | PathLike[str]):
        self.file_name = str(file_name)
        self.run_id = self._parse_run_id(Path(file_name).name)
        self.dl2_params = pd.read_hdf(self.file_name, key=dl2_params_lstcam_key)
        self._set_pointing()
        self.t_eff, self.t_elapsed = get_effective_time(self.dl2_params)
        self.dl2_provenance = read_dl2_provenance(self.file_name)
        self.analyze_provenance()

    @staticmethod
    def _parse_run_id(file_name: str) -> int:
        match = re.search(r"Run(?P<run_id>\d+)", file_name)
        if match is None:
            raise ValueError(f"could not find a run number in DL2 filename: {file_name}")
        return int(match.group("run_id"))

    def _set_pointing(self) -> None:
        if self.dl2_params.empty:
            raise ValueError(f"DL2 file contains no events: {self.file_name}")

        missing = self._required_pointing_columns.difference(self.dl2_params.columns)
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"DL2 data are missing required columns: {columns}")

        # lstchain uses the middle event because a numerical azimuth mean is
        # incorrect when a run crosses the 0/2π boundary.
        middle_event = self.dl2_params.iloc[len(self.dl2_params) // 2]
        self.pointing_alt = float(middle_event["alt_tel"])
        self.pointing_az = float(middle_event["az_tel"])
        self.pointing_zen = np.pi / 2 - self.pointing_alt

    @property
    def data(self):
        return self.dl2_params

    def analyze_provenance(self):
        self.input_path = None
        self.model_directory = None
        config = self.dl2_provenance.get("input", [])
        for input_info in config:
            roles = input_info.get("role", [])
            if "input" in roles:
                self.input_path = input_info.get("url")
            if "model" in roles:
                self.model_directory = input_info.get("url")

    @property
    def tailcut_level(self) -> tuple[int, int] | None:
        if self.input_path is None:
            return None
        match = re.search(r"tailcut(\d+)", self.input_path)
        if not match:
            return None
        digits = match.group(1)
        mid = len(digits) // 2
        return int(digits[:mid]), int(digits[mid:])

    @property
    def nsb_level(self) -> float | None:
        if self.model_directory is None:
            return None
        match = re.search(r"nsb_tuning_([\d.]+)", self.model_directory)
        return float(match.group(1)) if match else None

    @property
    def intensity_cuts(self) -> float:
        intensity_cuts = get_intensity_cut(self.data)
        return (intensity_cuts // 10 + 1) * 10

    @property
    def dec_line(self) -> int | None:
        if self.model_directory is None:
            return None
        match = re.search(r"dec_([\d.]+)", self.model_directory)
        return int(match.group(1)) if match else None
