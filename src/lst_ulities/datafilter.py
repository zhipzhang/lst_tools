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
    min_angle_to_source: float = 0.3
    max_angle_to_source: float = 0.5
    min_cos_zenith: float = 0
    max_cos_zenith: float = 1
    max_pointing_dec_std: float = 0.01  # degrees

    max_diffuse_nsb_std: float = 2.3
    max_intensity_at_half_peak_rate: float = 50
    min_mean_fit_p: float = -3
    min_drdi_index: float = -2.35
    max_drdi_index: float = -2.1
    min_drdi_at_422pe: float = 1.5
    min_fraction_around_mode: float = 0.8

    BASIC_CUTS = [
        "n_subruns",
        "date",
        "have_flatfield",
        "have_pedestal",
        "angle_to_source",
        "cos_zenith",
        "pointing_dec_std",
    ]

    def cut_masks(self, df: pd.DataFrame, advanced_cuts: bool = False) -> pd.DataFrame:
        """Return the active boolean cut masks.

        By default only the fundamental cuts are returned. Set ``advanced_cuts=True``
        to include the quality/physics cuts as well.
        """
        pointing = SkyCoord(ra=df["mean_ra"].to_numpy() * u.deg, dec=df["mean_dec"].to_numpy() * u.deg)  # pyright: ignore
        source = SkyCoord(ra=self.source_ra * u.deg, dec=self.source_dec * u.deg)  # pyright: ignore
        offset_angle = pointing.separation(source).to_value("deg")
        p_value_in_sigma = (df["mean_fit_p_value"] - 0.5) * np.sqrt(12 * df["n_subruns"])

        masks = pd.DataFrame(
            {
                "n_subruns": df["n_subruns"] > 0,
                "date": df["date"].between(self.first_date, self.last_date),
                "have_flatfield": df["n_flatfield"] >= 1,
                "have_pedestal": df["n_pedestals"] >= 1,
                "angle_to_source": (offset_angle >= self.min_angle_to_source)  # pyright: ignore
                & (offset_angle <= self.max_angle_to_source),  # pyright: ignore
                "cos_zenith": df["mean_cos_zd"].between(self.min_cos_zenith, self.max_cos_zenith),
                "pointing_dec_std": df["std_dec"] <= self.max_pointing_dec_std,
                "nsb_std": df["mean_diffuse_nsb_std"] <= self.max_diffuse_nsb_std,
                "intensity_threshold": ~(df["mean_intensity_threshold"] > self.max_intensity_at_half_peak_rate),
                "fit_p_value": p_value_in_sigma >= self.min_mean_fit_p,
                "drdi_index": df["mean_index"].between(self.min_drdi_index, self.max_drdi_index),
                "drdi_at_422pe": df["mean_R422"] >= self.min_drdi_at_422pe,
                "fraction_around_mode": df["fraction_around_mode_R422"] >= self.min_fraction_around_mode,
            },
            index=df.index,
        )
        return masks if advanced_cuts else masks[self.BASIC_CUTS]  # pyright: ignore

    def mask(self, df: pd.DataFrame, advanced_cuts: bool = False) -> pd.Series:
        """Return a boolean mask for the given run-statistics DataFrame."""
        return self.cut_masks(df, advanced_cuts=advanced_cuts).all(axis=1)  # pyright: ignore

    def cutflow(self, df: pd.DataFrame, advanced_cuts: bool = False) -> pd.Series:
        """Return the cumulative effect of the active cuts."""
        masks = self.cut_masks(df, advanced_cuts=advanced_cuts)
        remaining = masks.cumprod(axis=1).sum()
        return pd.concat([pd.Series({"total": len(df)}), remaining])

    def __call__(self, statistics: RunStatistics, advanced_cuts: bool = False) -> RunStatistics:
        """Apply the active cuts to a ``RunStatistics`` object.

        By default only the fundamental cuts are applied. Set ``advanced_cuts=True``
        to include the quality/physics cuts as well.
        """
        mask = self.mask(statistics.df, advanced_cuts=advanced_cuts)
        return statistics.select(mask)
