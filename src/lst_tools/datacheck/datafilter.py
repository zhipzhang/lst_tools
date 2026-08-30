from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

CRAB_NEBULA = SkyCoord.from_name("Crab Nebula")


@dataclass
class DataFilter:
    source_ra: float = CRAB_NEBULA.ra.to_value("deg")  # pyright: ignore
    source_dec: float = CRAB_NEBULA.dec.to_value("deg")  # pyright: ignore
    first_date: int = 0
    last_date: int = 29990101
    min_angle_to_source: float = 0.3
    max_angle_to_source: float = 0.5
    min_zenith_angle: float = 0
    max_zenith_angle: float = 90
    max_pointing_dec_std: float = 0.01  # degrees

    max_diffuse_nsb_std: float = 2.3
    max_intensity_at_half_peak_rate: float = 50
    min_mean_fit_p: float = -3
    min_drdi_index: float = -2.35
    max_drdi_index: float = -2.1
    min_drdi_at_422pe: float = 1.5
    min_fraction_around_mode: float = 0.8

    BASIC_COLUMNS = (
        "run_number",
        "n_subruns",
        "date",
        "n_flatfield",
        "n_pedestal",
        "mean_ra",
        "mean_dec",
        "mean_cos_zd",
        "pointing_dec_std",
    )
    ADVANCED_COLUMNS = (
        "mean_diffuse_nsb_std",
        "mean_intensity_threshold",
        "mean_fit_p_value",
        "mean_index",
        "mean_R422",
        "fraction_around_mode_R422",
    )

    def __call__(self, statistics: pd.DataFrame, advanced_cuts: bool = False) -> list[int]:
        """
        Apply the cuts on a `pd.DataFrame`
        and return the filtered list of run

        Parameters
        ----------
        statistics : pd.DataFrame
            The run-statistics DataFrame.
        advanced_cuts : bool, optional
            Whether to apply advanced cuts.

        Returns
        -------
        List[int]
            The filtered list of run numbers.
        """
        filtered = self.apply_basic_cuts(statistics)

        if advanced_cuts:
            filtered = self.apply_advanced_cuts(filtered)

        return filtered["run_number"].tolist()

    def apply_basic_cuts(self, statistics: pd.DataFrame) -> pd.DataFrame:
        """Return rows that pass all basic data-quality cuts."""
        if not set(self.BASIC_COLUMNS).issubset(statistics.columns):
            raise ValueError("BASIC_COLUMNS must exist as columns in statistics")

        pointing = SkyCoord(
            ra=statistics["mean_ra"].to_numpy() * u.Unit("deg"),
            dec=statistics["mean_dec"].to_numpy() * u.Unit("deg"),
        )
        source = SkyCoord(ra=self.source_ra * u.Unit("deg"), dec=self.source_dec * u.Unit("deg"))
        angle_to_source = pointing.separation(source).to_value("deg")

        min_cos_zenith = np.cos(np.radians(self.max_zenith_angle))
        max_cos_zenith = np.cos(np.radians(self.min_zenith_angle))

        mask = (
            statistics["n_subruns"].gt(0)
            & statistics["date"].between(self.first_date, self.last_date)
            & statistics["n_flatfield"].ge(1)
            & statistics["n_pedestal"].ge(1)
            & (angle_to_source >= self.min_angle_to_source)
            & (angle_to_source <= self.max_angle_to_source)
            & statistics["mean_cos_zd"].between(min_cos_zenith, max_cos_zenith)
            & statistics["pointing_dec_std"].le(self.max_pointing_dec_std)
        )
        return statistics.loc[mask]

    def apply_advanced_cuts(self, statistics: pd.DataFrame) -> pd.DataFrame:
        """Return rows that pass all advanced data-quality cuts."""
        if not set(self.ADVANCED_COLUMNS).issubset(statistics.columns):
            raise ValueError("ADVANCED_COLUMNS must exist as columns in statistics")
        if "n_subruns" not in statistics.columns:
            raise ValueError("n_subruns must exist as a column in statistics")

        p_value_in_sigma = (statistics["mean_fit_p_value"] - 0.5) * np.sqrt(12 * statistics["n_subruns"])

        mask = (
            statistics["mean_diffuse_nsb_std"].le(self.max_diffuse_nsb_std)
            & statistics["mean_intensity_threshold"].le(self.max_intensity_at_half_peak_rate)
            & p_value_in_sigma.ge(self.min_mean_fit_p)
            & statistics["mean_index"].between(self.min_drdi_index, self.max_drdi_index)
            & statistics["mean_R422"].ge(self.min_drdi_at_422pe)
            & statistics["fraction_around_mode_R422"].ge(self.min_fraction_around_mode)
        )
        return statistics.loc[mask]
