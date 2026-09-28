import re
from os import PathLike

import astropy.units as u
import numpy as np
from lstchain.io import read_simu_info_merged_hdf5
from numpy.typing import NDArray
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import CRAB_MAGIC_JHEAP2015, PowerLaw, calculate_event_weights

from .dl2table_base import LSTDL2TableBase


class LSTDL2MCTable(LSTDL2TableBase):
    """DL2 MC events loaded from one lstchain MC file."""

    def __init__(self, file_name: str | PathLike[str]):
        super().__init__(file_name)
        simu_config = read_simu_info_merged_hdf5(self.file_name)
        self.sim_info = SimulatedEventsInfo(
            n_showers=simu_config.n_showers * simu_config.shower_reuse,
            energy_min=simu_config.energy_range_min,
            energy_max=simu_config.energy_range_max,
            max_impact=simu_config.max_scatter_range,
            spectral_index=simu_config.spectral_index,
            viewcone_min=0 * u.deg,  # here, we enforce a zero viewcone for point simulation
            viewcone_max=0 * u.deg,
        )

    def calculate_weights_for_event(self, t_eff: u.Quantity, spectrum=CRAB_MAGIC_JHEAP2015) -> NDArray[np.float64]:
        simulated_spectrum = PowerLaw.from_simulation(self.sim_info, t_eff)
        mc_energy = self.dl2_params["mc_energy"].to_numpy() * u.TeV
        weights = calculate_event_weights(mc_energy, target_spectrum=spectrum, simulated_spectrum=simulated_spectrum)
        return weights

    @property
    def pointing_ze(self) -> float:
        match = re.search(r"node_theta_([\d.]+)", self.file_name)
        if match is None:
            raise ValueError(f"Could not parse pointing zenith from MC path: {self.file_name}")
        return float(match.group(1))

    @property
    def pointing_az(self) -> float:
        match = re.search(r"node_theta_[\d.]+_az_([\d.]+)", self.file_name)
        if match is None:
            raise ValueError(f"Could not parse pointing azimuth from MC path: {self.file_name}")
        return float(match.group(1))
