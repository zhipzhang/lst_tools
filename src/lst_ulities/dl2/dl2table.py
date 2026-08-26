import h5py
import hdf5plugin
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key
from pyirf.utils import calculate_theta

REDUCED_COLUMNS = (
    "x",
    "y",
    "r",
    "intensity",
    "log_intensity",
    "leakage_intensity_width_1",
    "leakage_intensity_width_2",
    "mc_energy",
    "reco_energy",
    "alt_tel",
    "az_tel",
    "reco_alt",
    "reco_az",
    "mc_alt",
    "mc_az",
)

rename_mapping = {
    "mc_energy": "true_energy",
    "reco_energy": "reco_energy",
    "mc_alt": "true_alt",
    "mc_az": "true_az",
    "reco_alt": "reco_alt",
    "reco_az": "reco_az",
    "alt_tel": "pointing_alt",
    "az_tel": "pointing_az",
}


class DL2McTable:
    def __init__(self, file_name: str, use_reduced=False):
        self.file_name = file_name
        self._file = h5py.File(file_name, "r")
        self._dl2_params = self._file[dl2_params_lstcam_key]

        if use_reduced:
            self.data = pd.DataFrame(self._dl2_params.fields(REDUCED_COLUMNS)[::], copy=False)
        else:
            self.data = pd.DataFrame(self._dl2_params[::], copy=False)
        self.data.rename(columns=rename_mapping, inplace=True)
        self.data["theta"] = calculate_theta(self.data, self.data["true_az"], self.data["true_alt"])
