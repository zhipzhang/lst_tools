"""Plot helpers for DL4 on/off count histograms."""

from typing import TYPE_CHECKING

import numpy as np
from hist import Hist, storage

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _get_axes(ax: "Axes | None") -> "Axes":
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    return ax


def _background_scale(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return alpha


def _excess_histogram(h_on: Hist, h_off: Hist, alpha: float) -> Hist:
    """Return an excess histogram with variance propagated by Hist."""
    required_axes = {"estimated_energy", "theta_squared"}
    for histogram in (h_on, h_off):
        names = {axis.name for axis in histogram.axes}
        if len(histogram.axes) != 2 or names != required_axes:
            raise ValueError("h_on and h_off must have exactly the 'estimated_energy' and 'theta_squared' axes")

    if any(on_axis != off_axis for on_axis, off_axis in zip(h_on.axes, h_off.axes, strict=True)):
        raise ValueError("h_on and h_off must use identical axes")

    if h_on.storage_type is not storage.Weight or h_off.storage_type is not storage.Weight:
        raise ValueError("h_on and h_off must track variances; use hist.storage.Weight")

    if h_on.variances(flow=True) is None or h_off.variances(flow=True) is None:
        raise ValueError("h_on and h_off must track variances; use hist.storage.Weight")

    try:
        excess = h_on.copy()
        excess_view = excess.view(flow=True)
        excess_view[...] = h_on.view(flow=True) - alpha * h_off.view(flow=True)
    except (TypeError, ValueError) as error:
        raise ValueError("h_on and h_off must use hist.storage.Weight") from error

    return excess


def plot_excess_counts(
    h_on: Hist,
    h_off: Hist,
    ax: "Axes | None" = None,
    alpha: float = 1.0,
    **kwargs,
) -> "Axes":
    """Plot background-subtracted counts versus estimated energy.

    Hist performs the on/off arithmetic and projects away the theta-squared
    axis. The excess and its Gaussian error approximation are

    ``N_on - alpha * N_off`` and ``sqrt(N_on + alpha**2 * N_off)``.

    Parameters
    ----------
    h_on, h_off : `hist.Hist`
        Filled two-dimensional on/off histograms.
    ax : `~matplotlib.axes.Axes`, optional
        Axes on which to draw. The current axes are used by default.
    alpha : float, optional
        Off-to-on normalization. Default is 1.
    **kwargs
        Additional arguments passed to `~matplotlib.axes.Axes.errorbar`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The axes containing the plot.
    """
    alpha = _background_scale(alpha)
    excess = _excess_histogram(h_on, h_off, alpha)
    excess_energy = excess.project("estimated_energy")
    energy_axis = excess_energy.axes[0]

    ax = _get_axes(ax)
    kwargs.setdefault("fmt", "o")
    kwargs.setdefault("capsize", 2)
    ax.errorbar(
        energy_axis.centers,
        excess_energy.values(flow=False),
        yerr=np.sqrt(excess_energy.variances(flow=False)),
        **kwargs,
    )
    if np.all(energy_axis.edges > 0):
        ax.set_xscale("log")
    ax.set_xlabel(energy_axis.label)
    ax.set_ylabel("Excess counts")
    return ax


def _interpolate_containment(
    excess: Hist,
    fraction: float,
) -> tuple[float, int, float, float]:
    """Interpolate a containment fraction linearly in theta."""
    excess_values = excess.values(flow=False)
    cumulative = np.cumsum(excess_values)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return np.nan, -1, np.nan, total

    target = fraction * total
    crossings = np.flatnonzero(cumulative >= target)
    if len(crossings) == 0:
        return np.nan, -1, np.nan, total

    index = int(crossings[0])
    count_before = cumulative[index - 1] if index else 0.0
    count_in_bin = excess_values[index]
    if count_in_bin <= 0:
        return np.nan, -1, np.nan, total

    bin_fraction = np.clip((target - count_before) / count_in_bin, 0.0, 1.0)
    theta_edges = np.sqrt(excess.axes[0].edges)
    theta = theta_edges[index] + bin_fraction * (theta_edges[index + 1] - theta_edges[index])
    return float(theta), index, float(bin_fraction), float(total)


def _containment_68(excess: Hist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_axis = excess.project("estimated_energy").axes[0]
    containment = np.full(energy_axis.size, np.nan)
    error_low = np.full(energy_axis.size, np.nan)
    error_high = np.full(energy_axis.size, np.nan)
    target_fraction = 0.68

    for energy_index in range(energy_axis.size):
        excess_theta = excess[{"estimated_energy": energy_index}]
        theta, index, bin_fraction, total = _interpolate_containment(excess_theta, target_fraction)
        if not np.isfinite(theta):
            continue

        variance_values = excess_theta.variances(flow=False)
        variance_total = excess_theta.sum(flow=False).variance
        variance_inside = variance_values[:index].sum() + bin_fraction * variance_values[index]
        fraction_variance = (
            variance_inside * (1 - 2 * target_fraction) + target_fraction**2 * variance_total
        ) / total**2
        fraction_sigma = np.sqrt(max(float(fraction_variance), 0.0))

        low_fraction = np.clip(target_fraction - fraction_sigma, 0.0, 1.0)
        high_fraction = np.clip(target_fraction + fraction_sigma, 0.0, 1.0)
        theta_low, _, _, _ = _interpolate_containment(excess_theta, low_fraction)
        theta_high, _, _, _ = _interpolate_containment(excess_theta, high_fraction)

        if np.isfinite(theta_low) and np.isfinite(theta_high):
            containment[energy_index] = theta
            error_low[energy_index] = max(theta - theta_low, 0.0)
            error_high[energy_index] = max(theta_high - theta, 0.0)

    return containment, error_low, error_high


def plot_excess_containment_68(
    h_on: Hist,
    h_off: Hist,
    ax: "Axes | None" = None,
    alpha: float = 1.0,
    **kwargs,
) -> "Axes":
    """Plot the interpolated 68% excess-containment radius versus energy.

    Hist performs the on/off arithmetic, named-axis slicing, and energy
    projection. NumPy is used only for the cumulative interpolation and final
    uncertainty arrays, for which Hist has no native operation.

    The asymmetric error bars are one-sigma approximations obtained by
    propagating the independent Poisson on/off variances into the cumulative
    excess fraction. Energy bins with non-positive total excess or undefined
    containment are omitted.

    Parameters
    ----------
    h_on, h_off : `hist.Hist`
        Filled two-dimensional on/off histograms.
    ax : `~matplotlib.axes.Axes`, optional
        Axes on which to draw. The current axes are used by default.
    alpha : float, optional
        Off-to-on normalization. Default is 1.
    **kwargs
        Additional arguments passed to `~matplotlib.axes.Axes.errorbar`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The axes containing the plot.
    """
    alpha = _background_scale(alpha)
    excess = _excess_histogram(h_on, h_off, alpha)
    energy_axis = excess.project("estimated_energy").axes[0]
    theta_squared_axis = excess.project("theta_squared").axes[0]
    if np.any(theta_squared_axis.edges < 0):
        raise ValueError("theta_squared bin edges must be non-negative")

    containment, error_low, error_high = _containment_68(excess)
    valid = np.isfinite(containment) & np.isfinite(error_low) & np.isfinite(error_high)

    ax = _get_axes(ax)
    kwargs.setdefault("fmt", "o")
    kwargs.setdefault("capsize", 2)
    ax.errorbar(
        energy_axis.centers[valid],
        containment[valid],
        yerr=np.vstack((error_low[valid], error_high[valid])),
        **kwargs,
    )
    if np.all(energy_axis.edges > 0):
        ax.set_xscale("log")
    ax.set_xlabel(energy_axis.label)
    ax.set_ylabel(r"68% excess containment radius [deg]")
    return ax
