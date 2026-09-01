import astropy.units as u
import matplotlib
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lst_tools.bkg.camera import CameraImage
from lst_tools.bkg.events import CameraEvents
from lst_tools.bkg.plot import plot_energy_slices, plot_radial_acceptance


@pytest.fixture
def center():
    return SkyCoord(alt=60 * u.deg, az=20 * u.deg, frame="altaz")


@pytest.fixture
def camera_events(center):
    return CameraEvents(
        center=center,
        x=[0.05, 0.1, 0.15, 0.3, 0.05, 0.1, 0.2, 0.4] * u.deg,
        y=[0, 0, 0, 0, 0, 0, 0, 0] * u.deg,
        energy=[0.2, 0.3, 0.5, 0.8, 2, 3, 5, 8] * u.TeV,
    )


def test_plot_energy_slices_creates_one_panel_per_energy_bin(center):
    image = CameraImage(
        center=center,
        x_edges=[-1, 0, 1],
        y_edges=[-1, 0, 1],
        e_edges=[0.1, 1, 10],
    )
    image.histogram.fill(
        x=[-0.5, 0.5, 0.5],
        y=[-0.5, 0.5, 0.5],
        energy=[0.5, 2, 3],
    )

    figure, axes = plot_energy_slices(image)

    assert isinstance(figure, Figure)
    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == ["0.1 ≤ E < 1 TeV", "1 ≤ E < 10 TeV"]
    assert [axis.collections[0].get_array().sum() for axis in axes] == [1, 2]
    assert all(axis.get_aspect() == 1 for axis in axes)
    plt.close(figure)


def test_plot_radial_acceptance_corrects_for_annular_solid_angle(camera_events):
    figure, axis = plot_radial_acceptance(
        camera_events,
        energy_edges=[0.1, 1, 10] * u.TeV,
        density=True,
    )

    assert isinstance(figure, Figure)
    assert len(axis.lines) == 2
    first, second = axis.lines
    np.testing.assert_allclose(first.get_xdata(), second.get_xdata())

    centers = first.get_xdata()
    radial_width = centers[1] - centers[0]
    radial_edges = np.concatenate(([centers[0] - radial_width / 2], centers + radial_width / 2))
    theta_edges = np.deg2rad(radial_edges)
    enclosed_solid_angle = 2 * np.pi * (1 - np.cos(theta_edges))
    annular_solid_angle = np.diff(enclosed_solid_angle)

    assert np.sum(first.get_ydata() * annular_solid_angle) == pytest.approx(1)
    assert np.sum(second.get_ydata() * annular_solid_angle) == pytest.approx(1)
    assert np.argmax(first.get_ydata()) == 0
    assert np.argmax(second.get_ydata()) == 0
    assert axis.get_ylabel() == "Probability density [sr⁻¹]"
    plt.close(figure)


def test_plot_radial_acceptance_can_show_event_density(camera_events):
    figure, axis = plot_radial_acceptance(
        camera_events,
        energy_edges=[0.1, 1, 10],
        density=False,
    )

    centers = axis.lines[0].get_xdata()
    radial_width = centers[1] - centers[0]
    radial_edges = np.concatenate(([centers[0] - radial_width / 2], centers + radial_width / 2))
    theta_edges = np.deg2rad(radial_edges)
    annular_solid_angle = np.diff(2 * np.pi * (1 - np.cos(theta_edges)))
    integrals = [np.sum(line.get_ydata() * annular_solid_angle) for line in axis.lines]

    np.testing.assert_allclose(integrals, [4, 4])
    assert axis.get_ylabel() == "Event density [sr⁻¹]"
    plt.close(figure)
