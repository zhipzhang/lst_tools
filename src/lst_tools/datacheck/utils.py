import numpy as np


def find_mode(data, binwidth=0.15, sliding_step=None, return_fraction=False):
    """
    return the bin center for the bin where contain most of the data.
    If return fraction is specified, return the fraction of the data in the mode bin.
    """
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan
    minimum = np.nanmin(data)
    maximum = np.nanmax(data)
    if np.isnan(minimum) or np.isnan(maximum):
        return np.nan
    if minimum == maximum:
        return np.nan

    if return_fraction and (maximum - minimum < binwidth):
        return 1

    while binwidth > (maximum - minimum):
        binwidth /= 2
    if sliding_step is None:
        sliding_step = binwidth / 100

    nn = int((maximum - minimum) // sliding_step + 2)
    cts, edges = np.histogram(
        data[~np.isnan(data)],
        bins=nn,
        range=(minimum - sliding_step, maximum + sliding_step),
    )
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


def azimuth_angle_mean(azimuth_angles):
    cos = np.cos(np.deg2rad(azimuth_angles))  # pyright: ignore
    sin = np.sin(np.deg2rad(azimuth_angles))
    mean_azimuth = np.arctan2(np.nanmean(sin), np.nanmean(cos))
    if mean_azimuth < 0:
        mean_azimuth += 2 * np.pi
    return np.degrees(mean_azimuth)
