import astropy.units as u
import numpy as np
import pytest

from lst_tools.bkg.camera import CameraImage
from lst_tools.bkg.events import SkyOffsetEvents


@pytest.fixture
def camera(observation_frame):
    return CameraImage(x_edges=[-1, 0, 1], y_edges=[-1, 0, 1], e_edges=[0.1, 1, 10], frame=observation_frame)


@pytest.fixture
def sample_events(observation_frame):
    return SkyOffsetEvents(
        x=[-0.5, 0.5, 2, 0, 0] * u.deg,
        y=[-0.5, 0.5, 0, 2, 0] * u.deg,
        energy=[0.5, 5, 0.5, 5, 20] * u.TeV,
        livetime=10 * u.s,
        frame=observation_frame,
    )


def test_camera_starts_with_empty_configured_axes(camera, observation_frame):
    assert camera.histogram.axes.name == ("x", "y", "energy")
    assert camera.frame is observation_frame
    assert camera.live_time == 0 * u.s
    np.testing.assert_array_equal(camera.histogram.values(), np.zeros((2, 2, 2)))


def test_from_events_bins_one_observation_and_copies_metadata(sample_events, observation_frame):
    image = CameraImage.from_events(sample_events, [-1, 0, 1], [-1, 0, 1], [0.1, 1, 10])

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 1
    expected[1, 1, 1] = 1
    np.testing.assert_array_equal(image.histogram.values(), expected)
    assert image.frame is observation_frame
    assert image.live_time == 10 * u.s


def test_from_events_accepts_empty_events(observation_frame):
    events = SkyOffsetEvents(frame=observation_frame)

    image = CameraImage.from_events(events, [-1, 1], [-1, 1], [0.1, 1])

    assert image.histogram.values().sum() == 0


def test_camera_has_no_cross_observation_fill(camera):
    assert not hasattr(camera, "fill")


def test_from_events_requires_sky_offset_events():
    with pytest.raises(TypeError, match="SkyOffsetEvents"):
        CameraImage.from_events(object(), [-1, 1], [-1, 1], [0.1, 1])


def test_to_radial_rebins_bin_centers_into_offset_and_energy(sample_events):
    image = CameraImage.from_events(sample_events, [-1, 0, 1], [-1, 0, 1], [0.1, 1, 10])

    radial = image.to_radial([0, 0.8, 1.2])

    assert radial.axes.name == ("energy", "theta")
    np.testing.assert_array_equal(radial.values(), [[1, 0], [1, 0]])


def test_to_radial_rejects_invalid_edges(camera):
    with pytest.raises(ValueError, match="theta_edges"):
        camera.to_radial([1, 0])


@pytest.mark.parametrize(
    ("edge_name", "edges"),
    [("x_edges", [0]), ("y_edges", [0, 0, 1]), ("e_edges", [0.1, np.nan, 1])],
)
def test_camera_rejects_invalid_axis_edges(observation_frame, edge_name, edges):
    axis_edges = {"x_edges": [-1, 0, 1], "y_edges": [-1, 0, 1], "e_edges": [0.1, 1, 10]}
    axis_edges[edge_name] = edges

    with pytest.raises(ValueError, match=edge_name):
        CameraImage(**axis_edges, frame=observation_frame)
