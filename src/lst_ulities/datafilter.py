from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .run_statistics import RunStatistics

CRAB_NEBULA = SkyCoord.from_name("Crab Nebula")


@dataclass
class DataFilter:
    source_ra: float = CRAB_NEBULA.ra.to_value("deg")  # pyright: ignore
    source_dec: float = CRAB_NEBULA.dec.to_value("deg")  # pyright: ignore
    first_date: int = 0
    last_date: int = 29990101
    max_diffuse_nsb_std: float = 2.3
    max_intensity_at_half_peak_rate: float = 50
    min_num_flatfield: int = 1
    min_num_pedestals: int = 1
    min_angle_to_source: float = 0.3
    max_angle_to_source: float = 0.5
    min_cos_zenith: float = 0
    max_cos_zenith: float = 1

    max_pointing_dec_std: float = 0.01  # degrees
    min_mean_fit_p: float = -3
    min_drdi_index: float = -2.35
    max_drdi_index: float = -2.1
    min_drdi_at_422pe: float = 1.5
    min_fraction_around_mode: float = 0.8

    def _angle_to_source(self, df: pd.DataFrame) -> pd.Series:
        pointing = SkyCoord(ra=df["mean_ra"].to_numpy() * u.deg, dec=df["mean_dec"].to_numpy() * u.deg)  # pyright: ignore
        source = SkyCoord(ra=self.source_ra * u.deg, dec=self.source_dec * u.deg)  # pyright: ignore
        return pd.Series(pointing.separation(source).to_value("deg"), index=df.index)

    def cut_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of boolean cut masks for the given run-statistics DataFrame."""
        offset_angle = self._angle_to_source(df)
        p_value_in_sigma = (df["mean_fit_p_value"] - 0.5) / np.sqrt(12 * df["n_subruns"])
        cuts = {
            "n_subruns": df["n_subruns"] > 0,
            "date": (df["date"] >= self.first_date) & (df["date"] <= self.last_date),
            "nsb_std": ~(df["mean_diffuse_nsb_std"] > self.max_diffuse_nsb_std),
            "intensity_threshold": ~(df["mean_intensity_threshold"] > self.max_intensity_at_half_peak_rate),
            "flatfield": df["n_flatfield"] >= self.min_num_flatfield,
            "pedestals": df["n_pedestals"] >= self.min_num_pedestals,
            "angle_to_source": (offset_angle >= self.min_angle_to_source) & (offset_angle <= self.max_angle_to_source),
            "cos_zenith": (df["mean_cos_zd"] >= self.min_cos_zenith) & (df["mean_cos_zd"] <= self.max_cos_zenith),
            "pointing_dec_std": df["std_dec"] <= self.max_pointing_dec_std,
            "fit_p_value": p_value_in_sigma >= self.min_mean_fit_p,
            "drdi_index": (df["mean_index"] >= self.min_drdi_index) & (df["mean_index"] <= self.max_drdi_index),
            "drdi_at_422pe": df["mean_R422"] >= self.min_drdi_at_422pe,
            "fraction_around_mode": df["fraction_around_mode_R422"] >= self.min_fraction_around_mode,
        }
        return pd.DataFrame(cuts, index=df.index)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask for the given run-statistics DataFrame."""
        return self.cut_masks(df).all(axis=1)  # pyright: ignore

    def cutflow(self, df: pd.DataFrame) -> pd.Series:
        remaining = pd.Series(True, index=df.index)
        out = {"total": int(remaining.sum())}
        for name, m in self.cut_masks(df).items():
            remaining &= m
            out[str(name)] = int(remaining.sum())
        return pd.Series(out)

    def __call__(self, statistics: RunStatistics) -> RunStatistics:
        mask = self.mask(statistics.df)
        return statistics.select(mask)
