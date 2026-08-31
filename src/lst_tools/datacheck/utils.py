from itertools import pairwise

import numpy as np
import pandas as pd


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
    sin = np.sin(np.deg2rad(azimuth_angles))  # pyright: ignore
    mean_azimuth = np.arctan2(np.nanmean(sin), np.nanmean(cos))
    if mean_azimuth < 0:
        mean_azimuth += 2 * np.pi
    return np.degrees(mean_azimuth)  # pyright: ignore


def validate_zenith_bin_edges(edges_deg) -> tuple[float, ...]:
    """Validate and normalize zenith-angle bin edges in degrees."""
    edges = np.asarray(edges_deg, dtype=float)

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("zenith binning requires at least two edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("zenith bin edges must be finite")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("zenith bin edges must be strictly increasing")
    if edges[0] < 0 or edges[-1] > 90:
        raise ValueError("zenith bin edges must be within [0, 90] degrees")

    return tuple(float(edge) for edge in edges)


def _format_zenith_edge(edge: float) -> str:
    """Format an edge for a filesystem-safe, readable bin name."""
    return f"{edge:g}".replace(".", "p")


def zenith_bin_labels(edges_deg) -> tuple[str, ...]:
    """Return stable labels for consecutive zenith-angle bin edges."""
    edges = validate_zenith_bin_edges(edges_deg)
    return tuple(f"zd_{_format_zenith_edge(low)}_{_format_zenith_edge(high)}" for low, high in pairwise(edges))


def assign_zenith_bins(statistics: pd.DataFrame, edges_deg) -> pd.DataFrame:
    """Return statistics with zenith-angle and zenith-bin columns added."""
    edges = validate_zenith_bin_edges(edges_deg)
    labels = zenith_bin_labels(edges)
    result = statistics.copy()
    zenith_angle = np.degrees(np.arccos(result["mean_cos_zd"].clip(-1.0, 1.0)))

    for edge in edges:
        zenith_angle = zenith_angle.mask(
            np.isclose(zenith_angle, edge, rtol=0.0, atol=1e-10),
            edge,
        )

    # ``right=False`` gives [low, high) bins. Move only the final edge by one
    # floating-point step so the configured upper boundary remains included.
    cut_edges = np.asarray(edges)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    assigned = pd.cut(
        zenith_angle,
        bins=cut_edges,
        labels=labels,
        right=False,
        ordered=True,
    )

    result["mean_zenith_angle"] = zenith_angle
    # Plain object strings are compatible with pandas/PyTables HDF5 output.
    result["zenith_bin"] = assigned.astype("object").where(assigned.notna(), None)
    return result
