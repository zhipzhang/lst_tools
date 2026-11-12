import numpy as np
import pandas as pd


def mean_fit_p_value_to_sigma(mean_p_value: pd.Series, n_subruns: pd.Series) -> pd.Series:
    """Convert mean per-subrun fit p-values to a sigma-equivalent deviation.

    For well-fitting subruns, the p-values follow a uniform U(0, 1) distribution.
    Their mean therefore has expectation 0.5 and standard deviation 1/sqrt(12*n),
    where n is the number of subruns. The returned value standardizes the observed
    mean using that expectation and standard deviation.

    Poor fits drive p-values toward 0, shifting their mean below 0.5 and producing
    negative values. A lower-tail cut such as -3 sigma rejects runs with strongly
    negative deviations while retaining the good-fit expectation around zero.
    """
    return (mean_p_value - 0.5) * np.sqrt(12 * n_subruns)


# NOTE: The cumulative-window indexing is intentionally unchanged pending a dedicated
# scientific review. Do not treat this as an independently validated mode estimator yet.
def sliding_window_mode(values, bin_width=0.15, sliding_step=None, return_fraction=False):
    """Estimate a mode center or its occupied fraction using a sliding histogram window.

    Empty, all-NaN, and constant inputs return NaN. If return_fraction is true and
    the value range is narrower than bin_width, the fraction is one.
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)
    if np.isnan(minimum) or np.isnan(maximum):
        return np.nan
    if minimum == maximum:
        return np.nan

    if return_fraction and (maximum - minimum < bin_width):
        return 1

    while bin_width > (maximum - minimum):
        bin_width /= 2
    if sliding_step is None:
        sliding_step = bin_width / 100

    n_bins = int((maximum - minimum) // sliding_step + 2)
    counts, edges = np.histogram(
        values[~np.isnan(values)],
        bins=n_bins,
        range=(minimum - sliding_step, maximum + sliding_step),
    )
    cumulative_fraction = np.cumsum(counts) / np.sum(counts)
    window_size = int(bin_width // sliding_step)
    running_fraction = cumulative_fraction[window_size:] - cumulative_fraction[:-window_size]
    window_centers = 0.5 * (edges[window_size:] + edges[:-window_size])[:-1]

    max_running_fraction = np.nanmax(running_fraction)
    if np.isnan(max_running_fraction):
        return np.nan
    if return_fraction:
        return max_running_fraction

    return window_centers[np.nanargmax(running_fraction)]


def circular_mean_degrees(angles) -> float:
    """Return the circular mean of angles in degrees, wrapped to [0, 360).

    NaNs are ignored. Empty and all-NaN inputs return NaN.
    """
    values = np.asarray(angles, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")

    mean_cos = np.nanmean(np.cos(np.deg2rad(values)))
    mean_sin = np.nanmean(np.sin(np.deg2rad(values)))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(np.degrees(mean_angle))
