"""Visualizations for camera images and event-level camera coordinates."""

from collections.abc import Iterable
from itertools import pairwise
from math import ceil, sqrt

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .camera import CameraImage, _validate_edges
from .events import CameraEvents


def _energy_label(low: float, high: float) -> str:
    return f"{low:g} ≤ E < {high:g} TeV"


def plot_energy_slices(
    image: CameraImage,
    *,
    max_columns: int = 3,
    cmap: str = "viridis",
) -> tuple[Figure, np.ndarray]:
    """Plot one sky-offset count map for every stored energy bin.

    Each panel uses its own color scale and color bar.
    """
    if not isinstance(max_columns, int) or max_columns < 1:
        raise ValueError("max_columns must be a positive integer")

    energy_bin_count = len(image.e_edges) - 1
    if energy_bin_count <= max_columns:
        column_count = energy_bin_count
    else:
        column_count = min(max_columns, ceil(sqrt(energy_bin_count)))
    row_count = ceil(energy_bin_count / column_count)
    figure, axes_grid = plt.subplots(
        row_count,
        column_count,
        squeeze=False,
        figsize=(4 * column_count, 3.6 * row_count),
        layout="constrained",
    )
    flat_axes = axes_grid.ravel()
    axes = flat_axes[:energy_bin_count]

    values = image.histogram.values()

    for energy_index, axis in enumerate(axes):
        slice_values = values[:, :, energy_index]
        maximum = float(np.max(slice_values, initial=0))
        normalization = Normalize(vmin=0, vmax=max(maximum, 1))
        mesh = axis.pcolormesh(
            image.x_edges,
            image.y_edges,
            slice_values.T,
            cmap=cmap,
            norm=normalization,
            shading="flat",
        )
        axis.set(
            title=_energy_label(image.e_edges[energy_index], image.e_edges[energy_index + 1]),
            xlabel="Sky-offset longitude [deg]",
            ylabel="Sky-offset latitude [deg]",
        )
        axis.set_aspect("equal")
        figure.colorbar(mesh, ax=axis, label="Events", shrink=0.9)

    for unused_axis in flat_axes[energy_bin_count:]:
        unused_axis.remove()

    return figure, axes


def plot_radial_acceptance(
    camera_events: CameraEvents,
    energy_edges: Iterable[float],
    *,
    density: bool = True,
    theta_squared: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot solid-angle-corrected radial acceptance by energy interval.

    A shared radial grid is inferred from all selected events so callers only
    choose the energy intervals. Counts in each radial interval are divided by
    its exact spherical solid angle. When ``density`` is true, each non-empty
    curve has unit solid-angle-weighted integral and can be compared
    independently of event count. When ``theta_squared`` is true, the
    horizontal axis shows the square of the camera offset radius.

    Exposure differences are not corrected by this function.
    """
    energy_bins = _validate_edges(energy_edges, "energy_edges", u.TeV)
    energy_intervals = list(pairwise(energy_bins))
    selected_events = [camera_events.select_energy(low, high) for low, high in energy_intervals]
    finite_radii = [
        radius[np.isfinite(radius)]
        for selected in selected_events
        if len(selected) > 0
        for radius in [selected.radius.to_value(u.deg)]
    ]
    all_radii = np.concatenate(finite_radii) if finite_radii else np.array([])
    if all_radii.size == 0:
        raise ValueError("no events fall inside the requested energy intervals")

    automatically_selected_edges = np.histogram_bin_edges(all_radii, bins="auto")
    # A line plot becomes noisy when an automatic estimator produces many
    # narrow annuli. Keep enough radial detail without drawing fluctuations as
    # structure in the acceptance.
    radial_bin_count = min(len(automatically_selected_edges) - 1, 25)
    radial_max = float(np.max(all_radii))
    if radial_max == 0:
        radial_max = 1.0
    radial_edges = np.linspace(0, radial_max, radial_bin_count + 1)
    if ax is None:
        figure, ax = plt.subplots(layout="constrained")
    else:
        figure = ax.figure

    theta_edges = np.deg2rad(radial_edges)
    enclosed_solid_angle = 2 * np.pi * (1 - np.cos(theta_edges))
    annular_solid_angle = np.diff(enclosed_solid_angle)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    horizontal_values = radial_centers**2 if theta_squared else radial_centers

    for (low, high), selected in zip(energy_intervals, selected_events, strict=True):
        radius = selected.radius.to_value(u.deg)
        radius = radius[np.isfinite(radius)]
        values, _ = np.histogram(radius, bins=radial_edges)
        values = values.astype(float)
        values /= annular_solid_angle
        if density and values.sum() > 0:
            values /= np.sum(values * annular_solid_angle)

        ax.plot(horizontal_values, values, label=_energy_label(low, high))

    ax.set(
        xlabel="Camera offset radius squared [deg²]" if theta_squared else "Camera offset radius [deg]",
        ylabel="Probability density [sr⁻¹]" if density else "Event density [sr⁻¹]",
    )
    ax.legend(title="Reconstructed energy")
    return figure, ax
