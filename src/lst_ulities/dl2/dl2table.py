import re
from typing import Optional

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance
from lstchain.reco.utils import get_effective_time, get_intensity_cut
from pyirf.utils import angular_separation

from ..datacheck import DataCheckStore, run_data_check


class LSTDL2EventTable:
    def __init__(self, file_name: str, data_check_store: None | DataCheckStore = None):
        self.file_name = file_name
        self.run_id = int(file_name.split("Run")[1].split(".")[0])
        if data_check_store is not None:
            self.data_check_store = data_check_store
        else:
            self.data_check_store = run_data_check

        if not self.data_check_store.is_initialized:
            raise RuntimeError("run_data_check is not initialized, using initialize_data_check method")
        if self.run_id not in self.data_check_store.run_statistics.run_numbers:
            raise RuntimeError(f"run_id {self.run_id} not found in run_statistics")
        self.pointing_ra = self.data_check_store.run_statistics.df.loc[self.run_id, "mean_ra"]  # degree
        self.pointing_dec = self.data_check_store.run_statistics.df.loc[self.run_id, "mean_dec"]  # degree
        self.run_check = self.data_check_store.data_check_tables.select_runs([self.run_id])
        self.pointing_alt = self.run_check.runsummary["mean_altitude"].iloc[0]  # rad
        self.pointing_az = self.run_check.runsummary["mean_azimuth"].iloc[0]  # rad

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
