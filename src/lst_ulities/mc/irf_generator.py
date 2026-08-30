import astropy.units as u
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyirf.irf.effective_area import create_histogram_table, effective_area
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import CRAB_MAGIC_JHEAP2015, PowerLaw, calculate_event_weights
from pyirf.utils import angular_separation, calculate_source_fov_offset

from .irf_binning import IRFBinning

MIN_GAMMANESS = 0.1  # This should be better configurable?


class IRFGenerator:
    def __init__(self, intensity_cut, gammaness_retain_ratio, irf_binning: IRFBinning):
        self.intensity_cuts = intensity_cut
        self.gammaness_retain_ratio = gammaness_retain_ratio
        self.irf_binning = irf_binning

    def __call__(self, data: pd.DataFrame, simulated_info: SimulatedEventsInfo):

        simulation_spectrum = PowerLaw.from_simulation(simulated_info, 50 * u.Unit("hour"))

        gammas = data.iloc[data["intensity"].values > self.intensity_cuts]
        reco_energy_bins = np.geomspace(
            self.irf_binning.reco_energy_min, self.irf_binning.reco_energy_max, self.irf_binning.reco_energy_bins + 1
        )
        true_energy_bins = np.geomspace(
            self.irf_binning.true_energy_min, self.irf_binning.true_energy_max, self.irf_binning.true_energy_bins + 1
        )
        self.gh_cuts = self.get_gammaness_cuts(1 - self.gammaness_retain_ratio, gammas, reco_energy_bins)

        gammas = self.apply_gammaness_cuts(gammas, self.gh_cuts, reco_energy_bins)
        gammas["true_source_fov_offset"] = angular_separation(
            gammas["true_az"], gammas["true_alt"], gammas["pointing_az"], gammas["pointing_alt"]
        )
        gammas["reco_source_fov_offset"] = angular_separation(
            gammas["reco_az"], gammas["reco_alt"], gammas["pointing_az"], gammas["pointing_alt"]
        )
        gammas["theta"] = angular_separation(
            gammas["true_az"], gammas["true_alt"], gammas["reco_az"], gammas["reco_alt"]
        )
        return self.effective_area(gammas, simulated_info, true_energy_bins)

    @staticmethod
    def get_gammaness_cuts(ratio: float, data: pd.DataFrame, reco_energy_bins: NDArray) -> pd.Series:
        gh_cuts = data.groupby(pd.cut(data["reco_energy"], bins=reco_energy_bins), observed=True, as_index=True)[
            "gammaness"
        ].quantile(ratio)
        return gh_cuts.clip(lower=MIN_GAMMANESS)

    @staticmethod
    def apply_gammaness_cuts(data: pd.DataFrame, gh_cuts: pd.Series, reco_energy_bins: NDArray):
        energy_bin: pd.Series = pd.cut(data["reco_energy"], bins=reco_energy_bins)
        event_cuts = energy_bin.map(gh_cuts).astype(float)

        return data.loc[event_cuts.notna() & (data["gammaness"] >= event_cuts)]

    @staticmethod
    def effective_area(
        data: pd.DataFrame,
        simulation_info: SimulatedEventsInfo,
        true_energy_bins: NDArray,
    ):
        true_energy = data["true_energy"].to_numpy()
        weights = data["weights"].to_numpy() if "weights" in data.columns else None
        hist_selected, _ = np.histogram(
            true_energy,
            bins=true_energy_bins,
            weights=weights,
        )
        area = np.pi * simulation_info.max_impact**2
        hist_simulated = simulation_info.calculate_n_showers_per_energy(true_energy_bins * u.Unit("TeV"))
        return effective_area(
            hist_selected,
            hist_simulated,
            area,
        )
