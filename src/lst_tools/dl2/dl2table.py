import re
from os import PathLike
from pathlib import Path

import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance
from lstchain.reco.utils import get_effective_time, get_intensity_cut


class LSTDL2EventTable:
    """DL2 events and run metadata loaded from one lstchain file."""

    def __init__(self, file_name: str | PathLike[str]):
        self.file_name = str(file_name)
        self.run_id = self._parse_run_id(Path(file_name).name)
        self.dl2_params: pd.DataFrame = pd.read_hdf(self.file_name, key=dl2_params_lstcam_key)
        self.t_eff, self.t_elapsed = get_effective_time(self.dl2_params)
        self.dl2_provenance = read_dl2_provenance(self.file_name)
        self.analyze_provenance()

    @staticmethod
    def _parse_run_id(file_name: str) -> int:
        match = re.search(r"Run(?P<run_id>\d+)", file_name)
        if match is None:
            raise ValueError(f"could not find a run number in DL2 filename: {file_name}")
        return int(match.group("run_id"))

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
        match = re.search(r"(?:^|/)dec_(?:(?P<negative>min)_)?(?P<digits>\d+)(?:/|$)", self.model_directory)
        if match is None:
            return None
        dec_line = int(match.group("digits"))
        return -dec_line if match.group("negative") else dec_line
