import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from lst_tools.bkg.camera import CameraImage


@pytest.fixture
def camera():
    pointing = SkyCoord(alt=60 * u.deg, az=20 * u.deg, frame="altaz")
    return CameraImage(
        center=pointing,
        x_edges=[-1, 0, 1],
        y_edges=[-1, 0, 1],
        e_edges=[0.1, 1, 10],
    )


@pytest.fixture
def sample_events(camera):
    """Two in-range events and one event outside each configured axis."""
    offsets = SkyCoord(
        lon=[-0.5, 0.5, 2, 0, 0] * u.deg,
        lat=[-0.5, 0.5, 0, 2, 0] * u.deg,
        frame=camera.telescope_frame,
    )
    horizontal = offsets.transform_to(camera.center.frame)

    return pd.DataFrame(
        {
            "reco_alt": horizontal.alt.to_value(u.deg),
            "reco_az": horizontal.az.to_value(u.deg),
            "reco_energy": [0.5, 5, 0.5, 5, 20],
        }
    )


def test_camera_skyoffsetframe(camera):
    event_pointing = SkyCoord(
        alt=camera.center.alt + 1 * u.deg,
        az=camera.center.az,
        frame=camera.center.frame,
    )
    transformed = event_pointing.transform_to(camera.telescope_frame)

    assert transformed.lon.to_value("deg") == pytest.approx(0)
    assert transformed.lat.to_value("deg") == pytest.approx(1)


def test_camera_starts_with_empty_configured_axes(camera):
    assert camera.histogram.axes.name == ("x", "y", "energy")
    np.testing.assert_allclose(camera.histogram.axes["x"].edges, [-1, 0, 1])
    np.testing.assert_allclose(camera.histogram.axes["y"].edges, [-1, 0, 1])
    np.testing.assert_allclose(camera.histogram.axes["energy"].edges, [0.1, 1, 10])
    np.testing.assert_array_equal(camera.histogram.values(), np.zeros((2, 2, 2)))


def test_fill_transforms_and_bins_events(camera, sample_events):
    camera.fill(sample_events)

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 1
    expected[1, 1, 1] = 1
    np.testing.assert_array_equal(camera.histogram.values(), expected)


def test_fill_accumulates_events(camera, sample_events):
    camera.fill(sample_events)
    camera.fill(sample_events)

    assert camera.histogram.values().sum() == 4


@pytest.mark.parametrize("missing_column", ["reco_alt", "reco_az", "reco_energy"])
def test_fill_requires_event_columns(camera, sample_events, missing_column):
    incomplete_events = sample_events.drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        camera.fill(incomplete_events)


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
            center=SkyCoord(alt=60 * u.deg, az=20 * u.deg, frame="altaz"),
            **axis_edges,
        )
