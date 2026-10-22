import re
from os import PathLike
from typing import cast

import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.io.provenance import read_dl2_provenance


class LSTDL2TableBase:
    """File access and provenance information of one DL2 table file."""

    def __init__(self, file_name: str | PathLike[str]):
        self.file_name = str(file_name)
        self.dl2_params: pd.DataFrame = self.read_dl2_events(self.file_name)
        self.dl2_provenance = read_dl2_provenance(self.file_name)
        self.dl1_file_path: str | None = None
        self.rf_model_directory: str | None = None
        self.analyze_provenance()

    @staticmethod
    def read_dl2_events(file_name: str) -> pd.DataFrame:
        """
        Read DL2 events from a HDF5 file.

        Adds ``gh_score`` as an alias of ``gammaness`` (the name lstchain
        uses for DL3/IRF work); both columns are kept.
        """
        events = cast(pd.DataFrame, pd.read_hdf(file_name, key=dl2_params_lstcam_key))
        if "gammaness" in events.columns and "gh_score" not in events.columns:
            events["gh_score"] = events["gammaness"]
        return events

    @property
    def data(self):
        return self.dl2_params

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
    def nsb_level(self) -> float | None:
        if self.rf_model_directory is None:
            return None
        match = re.search(r"nsb_tuning_([\d.]+)", self.rf_model_directory)
        return float(match.group(1)) if match else None

    @property
    def dec_line(self) -> int | None:
        if self.rf_model_directory is None:
            return None
        match = re.search(r"(?:^|/)dec_(?:(?P<negative>min)_)?(?P<digits>\d+)(?:/|$)", self.rf_model_directory)
        if match is None:
            return None
        dec_line = int(match.group("digits"))
        return -dec_line if match.group("negative") else dec_line
