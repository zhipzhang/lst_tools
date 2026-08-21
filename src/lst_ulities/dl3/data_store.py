"""
A small helper class for generating `gammapy.DataStore`.

Main features:
    - Automatically Generate the index file
    - Able to merge multiple data stores
"""

import logging
from pathlib import Path

import numpy as np
from gammapy.data import DataStore, Observation
from gammapy.maps import MapAxis

from lst_ulities.helper import init_plot


def generate_index_file(path):
    from shutil import which

    if which("lstchain_create_dl3_index_files") is None:
        raise RuntimeError(
            "lstchain_create_dl3_index_files is not available in PATH. "
            "Please install lstchain or add its executable to PATH."
        )
    import subprocess

    subprocess.run(["lstchain_create_dl3_index_files", "-d", path, "-o", path], check=True)


class Dl3DataStore:
    def __init__(self, path):
        if not Path(f"{path}/obs-index.fits.gz").exists():
            logging.warning(f"No index file found at {path}, try to generate it")
            generate_index_file(path)

        self.data_store = DataStore.from_dir(path)

    def __getattr__(self, name):
        return getattr(self.data_store, name)

    def plot_1d_angres(self):
        """
        Plot the observation-time-averaged angular resolution versus energy.
        """
        import matplotlib.pyplot as plt

        init_plot()
        try:
            observations = self.data_store.get_observations(required_irf="point-like")
        except Exception as e:
            logging.error(f"1D angular resolution plot must be performed with point-like IRFs: {e}")
            return

        energy_axis: MapAxis | None = None
        angular_resolution = []
        effective_obs_time = []
        for obs in observations:
            if energy_axis is None:
                energy_axis = obs.rad_max.axes[0]
            angular_resolution.append(np.ravel(obs.rad_max.data))
            effective_obs_time.append(obs.observation_live_time_duration.to_value("s"))

        if energy_axis is None:
            logging.warning("No observations available to plot angular resolution")
            return

        angular_resolution = np.asarray(angular_resolution)
        effective_obs_time = np.asarray(effective_obs_time)
        obs_time_averaged_angres = np.average(angular_resolution, axis=0, weights=effective_obs_time)
        energy_centers = energy_axis.center.to_value("TeV")

        plt.plot(energy_centers, obs_time_averaged_angres)
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Angular resolution (deg)")
        plt.xscale("log")
        plt.show()
