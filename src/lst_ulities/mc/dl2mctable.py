from pathlib import Path

import h5py
import hdf5plugin
import pandas as pd
from lstchain.io.io import dl2_params_lstcam_key, read_simu_info_merged_hdf5
from pyirf.simulations import SimulatedEventsInfo

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
    "disp_sign",
    "reco_disp_sign",
    "gammaness",
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


class LSTDL2MCTable:
    def __init__(self, file_path: str | Path, use_reduced_columns: bool = True):
        self.file_path = Path(file_path)
        with h5py.File(self.file_path, "r") as f:
            dl2_params = f[dl2_params_lstcam_key]
            if use_reduced_columns:
                self.dl2_params = pd.DataFrame(dl2_params.fields(REDUCED_COLUMNS)[::], copy=False)
            else:
                self.dl2_params = pd.DataFrame(dl2_params[:], copy=False)

            self.dl2_params.rename(columns=rename_mapping, inplace=True)
        simu_info = read_simu_info_merged_hdf5(self.file_path)
        self.simu_info = SimulatedEventsInfo(
            n_showers=simu_info.n_showers * simu_info.shower_reuse,
            energy_min=simu_info.energy_range_min,
            energy_max=simu_info.energy_range_max,
            max_impact=simu_info.max_scatter_range,
            spectral_index=simu_info.spectral_index,
            viewcone_min=simu_info.min_viewcone_radius,
            viewcone_max=simu_info.max_viewcone_radius,
        )

    @property
    def data(self):
        return self.dl2_params
