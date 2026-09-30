import astropy.units as u
import numpy as np
import pytest

from lst_tools.bkg.camera import CameraImage
from lst_tools.bkg.events import SkyOffsetEvents


@pytest.fixture
def camera():
    return CameraImage(
        x_edges=[-1, 0, 1],
        y_edges=[-1, 0, 1],
        e_edges=[0.1, 1, 10],
    )


@pytest.fixture
def sample_events():
    """Two in-range events and one event outside each configured axis."""
    return SkyOffsetEvents(
        x=[-0.5, 0.5, 2, 0, 0] * u.deg,
        y=[-0.5, 0.5, 0, 2, 0] * u.deg,
        energy=[0.5, 5, 0.5, 5, 20] * u.TeV,
        livetime=10 * u.s,
    )


def test_camera_starts_with_empty_configured_axes(camera):
    assert camera.histogram.axes.name == ("x", "y", "energy")
    np.testing.assert_allclose(camera.histogram.axes["x"].edges, [-1, 0, 1])
    np.testing.assert_allclose(camera.histogram.axes["y"].edges, [-1, 0, 1])
    np.testing.assert_allclose(camera.histogram.axes["energy"].edges, [0.1, 1, 10])
    np.testing.assert_array_equal(camera.histogram.values(), np.zeros((2, 2, 2)))


def test_fill_bins_sky_offset_events(camera, sample_events):
    camera.fill(sample_events)

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 1
    expected[1, 1, 1] = 1
    np.testing.assert_array_equal(camera.histogram.values(), expected)


def test_fill_accumulates_events(camera, sample_events):
    camera.fill(sample_events)
    camera.fill(sample_events)

    assert camera.histogram.values().sum() == 4


def test_fill_accepts_empty_events(camera):
    camera.fill(SkyOffsetEvents())

    assert camera.histogram.values().sum() == 0


def test_fill_requires_sky_offset_events(camera):
    with pytest.raises(TypeError, match="SkyOffsetEvents"):
        camera.fill(object())


@pytest.mark.parametrize(
    ("edge_name", "edges"),
    [
        ("x_edges", [0]),
        ("y_edges", [0, 0, 1]),
        ("e_edges", [0.1, np.nan, 1]),
    ],
)
def test_camera_rejects_invalid_axis_edges(edge_name, edges):
    axis_edges = {
        "x_edges": [-1, 0, 1],
        "y_edges": [-1, 0, 1],
        "e_edges": [0.1, 1, 10],
    }
    axis_edges[edge_name] = edges

    with pytest.raises(ValueError, match=edge_name):
        CameraImage(
            **axis_edges,
        )
