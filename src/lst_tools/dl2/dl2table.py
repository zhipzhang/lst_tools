import re

import numpy as np
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance
from lstchain.reco.utils import get_effective_time, get_intensity_cut


class LSTDL2EventTable:
    def __init__(self, file_name: str, statistics: pd.DataFrame):
        self.file_name = file_name
        self.run_id = int(file_name.split("Run")[1].split(".")[0])
        run = statistics.loc[statistics["run_number"].eq(self.run_id)]
        if run.empty:
            raise RuntimeError(f"run_id {self.run_id} not found in statistics")
        if len(run) > 1:
            raise RuntimeError(f"run_id {self.run_id} occurs more than once in statistics")

        self.pointing_ra = run["mean_ra"].iloc[0]
        self.pointing_dec = run["mean_dec"].iloc[0]
        self.pointing_zen = np.arccos(run["mean_cos_zd"].iloc[0])
        self.dl2_params = pd.read_hdf(self.file_name, key=dl2_params_lstcam_key)
        self.t_eff, self.t_elapsed = get_effective_time(self.dl2_params)
        self.dl2_provenance = read_dl2_provenance(self.file_name)
        self.analyze_provenance()

    @property
    def data(self):
        return self.dl2_params

    def analyze_provenance(self):
        config = self.dl2_provenance["input"]
        for input_info in config:
            if "input" in input_info["role"]:
                self.input_path = input_info["url"]
            if "model" in input_info["role"]:
                self.model_directory = input_info["url"]

    @property
    def tailcut_level(self) -> tuple[int, int] | None:
        match = re.search(r"tailcut(\d+)", self.input_path)
        if not match:
            return None
        digits = match.group(1)
        mid = len(digits) // 2
        return int(digits[:mid]), int(digits[mid:])

    @property
    def nsb_level(self) -> float | None:
        match = re.search(r"nsb_tuning_([\d.]+)", self.model_directory)
        return float(match.group(1)) if match else None

    @property
    def intensity_cuts(self) -> float:
        intensity_cuts = get_intensity_cut(self.data)
        return (intensity_cuts // 10 + 1) * 10

    @property
    def dec_line(self) -> int | None:
        match = re.search(r"dec_([\d.]+)", self.model_directory)
        return int(match.group(1)) if match else None
