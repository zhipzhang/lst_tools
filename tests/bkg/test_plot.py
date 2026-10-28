import astropy.units as u
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lst_tools.bkg.camera import CameraImage
from lst_tools.bkg.events import SkyOffsetEvents
from lst_tools.bkg.plot import plot_energy_slices, plot_radial_acceptance

THETA_EDGES = np.linspace(0, 0.5, 6)


@pytest.fixture
def sky_offset_events(observation_frame):
    return SkyOffsetEvents(
        x=[0.05, 0.1, 0.15, 0.3, 0.05, 0.1, 0.2, 0.4] * u.deg,
        y=[0, 0, 0, 0, 0, 0, 0, 0] * u.deg,
        energy=[0.2, 0.3, 0.5, 0.8, 2, 3, 5, 8] * u.TeV,
        livetime=10 * u.s,
        frame=observation_frame,
    )


def test_plot_energy_slices_creates_one_panel_per_energy_bin(observation_frame):
    image = CameraImage(
        x_edges=[-1, 0, 1],
        y_edges=[-1, 0, 1],
        e_edges=[0.1, 1, 10],
        frame=observation_frame,
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


def test_plot_radial_acceptance_corrects_for_annular_solid_angle(sky_offset_events):
    figure, axis = plot_radial_acceptance(
        sky_offset_events,
        energy_edges=[0.1, 1, 10] * u.TeV,
        theta_edges=THETA_EDGES * u.deg,
        density=True,
    )

    assert isinstance(figure, Figure)
    assert len(axis.lines) == 2
    first, second = axis.lines
    np.testing.assert_allclose(first.get_xdata(), second.get_xdata())

    centers = 0.5 * (THETA_EDGES[:-1] + THETA_EDGES[1:])
    np.testing.assert_allclose(first.get_xdata(), centers)
    theta_edges = np.deg2rad(THETA_EDGES)
    enclosed_solid_angle = 2 * np.pi * (1 - np.cos(theta_edges))
    annular_solid_angle = np.diff(enclosed_solid_angle)

    assert np.sum(first.get_ydata() * annular_solid_angle) == pytest.approx(1)
    assert np.sum(second.get_ydata() * annular_solid_angle) == pytest.approx(1)
    assert np.argmax(first.get_ydata()) == 0
    assert np.argmax(second.get_ydata()) == 0
    assert axis.get_ylabel() == "Probability density [sr⁻¹]"
    plt.close(figure)


def test_plot_radial_acceptance_can_show_event_density(sky_offset_events):
    figure, axis = plot_radial_acceptance(
        sky_offset_events,
        energy_edges=[0.1, 1, 10],
        theta_edges=THETA_EDGES,
        density=False,
    )

    theta_edges = np.deg2rad(THETA_EDGES)
    annular_solid_angle = np.diff(2 * np.pi * (1 - np.cos(theta_edges)))
    integrals = [np.sum(line.get_ydata() * annular_solid_angle) for line in axis.lines]

    np.testing.assert_allclose(integrals, [4, 4])
    assert axis.get_ylabel() == "Event density [sr⁻¹]"
    plt.close(figure)


def test_plot_radial_acceptance_can_plot_an_empty_sample(observation_frame):
    figure, axis = plot_radial_acceptance(
        SkyOffsetEvents(frame=observation_frame),
        energy_edges=[0.1, 1, 10],
        theta_edges=THETA_EDGES,
    )

    assert len(axis.lines) == 2
    assert all(np.all(line.get_ydata() == 0) for line in axis.lines)
    plt.close(figure)


def test_plot_radial_acceptance_rejects_event_collections(sky_offset_events):
    with pytest.raises(TypeError, match="SkyOffsetEvents"):
        plot_radial_acceptance(
            [sky_offset_events],
            energy_edges=[0.1, 1, 10],
            theta_edges=THETA_EDGES,
        )


def test_plot_radial_acceptance_can_show_theta_squared(sky_offset_events):
    figure, axis = plot_radial_acceptance(
        sky_offset_events,
        energy_edges=[0.1, 1, 10],
        theta_edges=THETA_EDGES,
        theta_squared=True,
    )

    centers = 0.5 * (THETA_EDGES[:-1] + THETA_EDGES[1:])
    np.testing.assert_allclose(axis.lines[0].get_xdata(), centers**2)
    assert axis.get_xlabel() == "Camera offset radius squared [deg²]"
    plt.close(figure)


@pytest.mark.parametrize("theta_edges", ([-0.1, 0, 1], [0, 90, 181]))
def test_plot_radial_acceptance_rejects_unphysical_theta_edges(sky_offset_events, theta_edges):
    with pytest.raises(ValueError, match="between 0 and 180"):
        plot_radial_acceptance(
            sky_offset_events,
            energy_edges=[0.1, 1, 10],
            theta_edges=theta_edges,
        )
