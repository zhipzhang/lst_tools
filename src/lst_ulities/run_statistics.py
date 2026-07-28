from dataclasses import dataclass
from functools import reduce

import astropy.units as u
import numpy as np
import pandas as pd


def find_mode(data, binwidth=0.15, sliding_step=None, return_fraction=False):
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan
    min = np.nanmin(data)
    max = np.nanmax(data)
    if np.isnan(min) or np.isnan(max):
        return np.nan
    if min == max:
        return np.nan

    if return_fraction and (max - min < binwidth):
        return 1

    while binwidth > (max - min):
        binwidth /= 2
    if sliding_step is None:
        sliding_step = binwidth / 100

    nn = int((max - min) // sliding_step + 2)
    cts, edges = np.histogram(data[~np.isnan(data)], bins=nn, range=(min - sliding_step, max + sliding_step))
    csum = np.cumsum(cts) / np.sum(cts)
    nsumbins = int(binwidth // sliding_step)
    running_sum = csum[nsumbins:] - csum[:-nsumbins]
    xvalues = 0.5 * (edges[nsumbins:] + edges[:-nsumbins])[:-1]

    max_running_sum = np.nanmax(running_sum)
    if np.isnan(max_running_sum):
        return np.nan
    if return_fraction:
        return max_running_sum

    return xvalues[np.nanargmax(running_sum)]


def zenith_angle_mean(zenith_angles):
    cos = np.cos(zenith_angles * u.deg)  # pyright: ignore
    sin = np.sin(zenith_angles * u.deg)  # pyright: ignore
    mean_zenith = np.arctan2(np.nanmean(sin), np.nanmean(cos))
    if mean_zenith < 0:
        mean_zenith += 2 * np.pi
    return np.degrees(mean_zenith)  # pyright: ignore


DEFAULT_SPEC = {
    "cosmics_intensity_spectrum": {
        "n_subruns": ("runnumber", "size"),
        "date": ("yyyymmdd", "first"),
        "mean_R422": ("ZD_corrected_cosmics_rate_at_422_pe", "mean"),
        "std_R422": ("ZD_corrected_cosmics_rate_at_422_pe", "std"),
        "mode_R422": ("ZD_corrected_cosmics_rate_at_422_pe", lambda x: find_mode(x)),
        "fraction_around_mode_R422": (
            "ZD_corrected_cosmics_rate_at_422_pe",
            lambda x: find_mode(x, return_fraction=True),
        ),
        "mean_intensity_at_reference_rate": ("intensity_at_reference_rate", "mean"),
        "std_intensity_at_reference_rate": ("intensity_at_reference_rate", "std"),
        "mean_light_yield": ("light_yield", "mean"),
        "std_light_yield": ("light_yield", "std"),
        "mean_index": ("ZD_corrected_cosmics_spectral_index", "mean"),
        "std_index": ("ZD_corrected_cosmics_spectral_index", "std"),
        "mean_fit_p_value": ("intensity_spectrum_fit_p_value", "mean"),
        "mean_intensity_threshold": ("ZD_corrected_intensity_at_half_peak_rate", "mean"),
        "std_intensity_threshold": ("ZD_corrected_intensity_at_half_peak_rate", "std"),
        "mean_ra": ("ra_tel", lambda x: zenith_angle_mean(x)),
        "mean_dec": ("dec_tel", "mean"),
        "std_dec": ("dec_tel", "std"),
        "mean_cos_zd": ("cos_zenith", "mean"),
        "mean_diffuse_nsb_std": ("diffuse_nsb_std", "mean"),
    },
    "runsummary": {
        "n_flatfield": ("num_flatfield", "first"),
        "n_pedestals": ("num_pedestals", "first"),
    },
}


@dataclass
class RunStatistics:
    df: pd.DataFrame

    @classmethod
    def from_tables(cls, tables, spec=DEFAULT_SPEC) -> "RunStatistics":
        per_table = []
        for table_name, agg_spec in spec.items():
            df = getattr(tables, table_name)
            per_table.append(df.groupby("runnumber").agg(**agg_spec))
        ## Outer join on runnumber
        merged = reduce(lambda a, b: a.join(b, how="outer"), per_table)
        return cls(merged.sort_index())

    def select(self, mask: pd.Series) -> "RunStatistics":
        return RunStatistics(self.df.loc[mask])

    @property
    def run_numbers(self) -> pd.Index:
        return self.df.index
