import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from lst_tools.bkg.events import CameraEvents

from lst_tools.bkg.camera import CameraImage


@pytest.fixture
def pointing():
    return SkyCoord(alt=60 * u.deg, az=20 * u.deg, frame="altaz")


@pytest.fixture
def events(pointing):
    """Events at known camera offsets, including one radial outlier."""
    offset_frame = SkyOffsetFrame(origin=pointing)
    offsets = SkyCoord(
        lon=[-0.8, -0.2, 0.2, 0.8, 0, 2] * u.deg,
        lat=[0, 0, 0, 0, 0, 0] * u.deg,
        frame=offset_frame,
    )
    horizontal = offsets.transform_to(pointing.frame)

    return pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13, 14, 15],
            "reco_alt": horizontal.alt.to_value(u.deg),
            "reco_az": horizontal.az.to_value(u.deg),
            "reco_energy": [0.2, 0.8, 1.5, 6, 3, 2],
            "gammaness": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        },
        index=[5, 3, 8, 1, 9, 4],
    )


@pytest.fixture
def camera_events(events, pointing):
    return CameraEvents.from_dataframe(events, center=pointing)


def test_from_dataframe_preserves_event_level_coordinates(camera_events):
    assert len(camera_events) == 6
    assert camera_events.x.unit == u.deg
    assert camera_events.y.unit == u.deg
    assert camera_events.energy.unit == u.TeV
    np.testing.assert_allclose(camera_events.x.to_value(u.deg), [-0.8, -0.2, 0.2, 0.8, 0, 2], atol=1e-12)
    np.testing.assert_allclose(camera_events.y.to_value(u.deg), 0, atol=1e-12)
    np.testing.assert_allclose(camera_events.energy.to_value(u.TeV), [0.2, 0.8, 1.5, 6, 3, 2])


def test_from_dataframe_does_not_modify_input(events, pointing):
    original = events.copy(deep=True)

    CameraEvents.from_dataframe(events, center=pointing)

    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize("missing_column", ["reco_alt", "reco_az", "reco_energy"])
def test_from_dataframe_requires_event_columns(events, pointing, missing_column):
    incomplete_events = events.drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        CameraEvents.from_dataframe(incomplete_events, center=pointing)


def test_to_image_uses_requested_bins(camera_events):
    image = camera_events.to_image(
        x_edges=[-1, 0, 1] * u.deg,
        y_edges=[-0.5, 0.5] * u.deg,
        e_edges=[0.1, 1, 10] * u.TeV,
    )

    assert isinstance(image, CameraImage)
    expected = np.zeros((2, 1, 2))
    expected[0, 0, 0] = 2
    expected[1, 0, 1] = 3
    np.testing.assert_array_equal(image.histogram.values(), expected)


def test_to_image_can_apply_different_energy_bins(camera_events):
    coarse = camera_events.to_image(
        x_edges=[-1, 1],
        y_edges=[-1, 1],
        e_edges=[0.1, 1, 10],
    )
    fine = camera_events.to_image(
        x_edges=[-1, 1],
        y_edges=[-1, 1],
        e_edges=[0.1, 0.5, 2, 10],
    )

    np.testing.assert_array_equal(coarse.histogram.values()[0, 0], [2, 3])
    np.testing.assert_array_equal(fine.histogram.values()[0, 0], [1, 2, 2])


def test_radius_remains_available_for_each_event(camera_events):
    assert camera_events.radius.unit == u.deg
    np.testing.assert_allclose(camera_events.radius.to_value(u.deg), [0.8, 0.2, 0.2, 0.8, 0, 2], atol=1e-12)


def test_select_energy_preserves_events_in_requested_interval(camera_events):
    selected = camera_events.select_energy(0.5 * u.TeV, 2 * u.TeV)

    # Energy intervals include the lower edge and exclude the upper edge.
    np.testing.assert_allclose(selected.energy.to_value(u.TeV), [0.8, 1.5])
    np.testing.assert_allclose(selected.x.to_value(u.deg), [-0.2, 0.2], atol=1e-12)
    np.testing.assert_allclose(selected.radius.to_value(u.deg), [0.2, 0.2], atol=1e-12)


def test_select_energy_accepts_unitless_tev_bounds(camera_events):
    selected = camera_events.select_energy(2, 10)

    np.testing.assert_allclose(selected.energy.to_value(u.TeV), [6, 3, 2])
