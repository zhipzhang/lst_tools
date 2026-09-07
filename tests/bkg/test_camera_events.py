import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import AltAz, SkyCoord, SkyOffsetFrame
from astropy.time import Time
from astropy.utils import iers

from lst_tools.bkg.camera import CameraImage
from lst_tools.bkg.events import LST_LOCATION, CameraEvents, _mean_direction


@pytest.fixture(scope="module", autouse=True)
def use_bundled_iers_data():
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.fixture
def pointing():
    return SkyCoord(ra=83 * u.deg, dec=22 * u.deg, frame="icrs")


@pytest.fixture
def events(pointing):
    """DL2 events at known sky offsets, including one radial outlier."""
    event_time = Time(1_700_000_000 + np.arange(6) * 60, format="unix")
    altaz_frame = AltAz(obstime=event_time, location=LST_LOCATION)
    offset_frame = SkyOffsetFrame(origin=pointing)
    offsets = SkyCoord(
        lon=[-0.8, -0.2, 0.2, 0.8, 0, 2] * u.deg,
        lat=[0, 0, 0, 0, 0, 0] * u.deg,
        frame=offset_frame,
    )
    reconstructed = offsets.transform_to(altaz_frame)
    telescope_pointing = pointing.transform_to(altaz_frame)

    return pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13, 14, 15],
            "reco_alt": reconstructed.alt.to_value(u.rad),
            "reco_az": reconstructed.az.to_value(u.rad),
            "reco_energy": [0.2, 0.8, 1.5, 6, 3, 2],
            "alt_tel": telescope_pointing.alt.to_value(u.rad),
            "az_tel": telescope_pointing.az.to_value(u.rad),
            "trigger_time": event_time.unix,
            "gammaness": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        },
        index=[5, 3, 8, 1, 9, 4],
    )


@pytest.fixture
def camera_events(events):
    return CameraEvents.from_dataframe(events)


def test_from_dataframe_preserves_event_level_coordinates(camera_events, pointing):
    assert len(camera_events) == 6
    assert camera_events.center.separation(pointing) < 1e-6 * u.deg
    assert camera_events.x.unit == u.deg
    assert camera_events.y.unit == u.deg
    assert camera_events.energy.unit == u.TeV
    np.testing.assert_allclose(camera_events.x.to_value(u.deg), [-0.8, -0.2, 0.2, 0.8, 0, 2], atol=1e-6)
    np.testing.assert_allclose(camera_events.y.to_value(u.deg), 0, atol=1e-6)
    np.testing.assert_allclose(camera_events.energy.to_value(u.TeV), [0.2, 0.8, 1.5, 6, 3, 2])


def test_from_dataframe_does_not_modify_input(events):
    original = events.copy(deep=True)

    CameraEvents.from_dataframe(events)

    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize("missing_column", CameraEvents.REQUIRED_COLUMNS)
def test_from_dataframe_requires_event_columns(events, missing_column):
    incomplete_events = events.drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        CameraEvents.from_dataframe(incomplete_events)


def test_from_dataframe_rejects_empty_events(events):
    with pytest.raises(ValueError, match="at least one event"):
        CameraEvents.from_dataframe(events.iloc[:0])


def test_mean_direction_handles_ra_wrap():
    directions = SkyCoord(ra=[359.9, 0.1] * u.deg, dec=[20, 20] * u.deg, frame="icrs")
    expected_center = SkyCoord(ra=0 * u.deg, dec=20 * u.deg, frame="icrs")

    center = _mean_direction(directions)

    assert center.separation(expected_center) < 1 * u.arcsec


def test_center_must_be_scalar():
    centers = SkyCoord(ra=[0, 1] * u.deg, dec=[0, 1] * u.deg, frame="icrs")

    with pytest.raises(ValueError, match="scalar SkyCoord"):
        CameraEvents(center=centers, x=[0, 1], y=[0, 1], energy=[1, 1])


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
    np.testing.assert_allclose(camera_events.radius.to_value(u.deg), [0.8, 0.2, 0.2, 0.8, 0, 2], atol=1e-6)


def test_radius_is_an_exact_spherical_separation(pointing):
    camera_events = CameraEvents(
        center=pointing,
        x=[1] * u.deg,
        y=[1] * u.deg,
        energy=[1] * u.TeV,
    )
    event = SkyCoord(ra=1 * u.deg, dec=1 * u.deg, frame="icrs")
    center = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs")

    assert camera_events.radius[0] == event.separation(center)


def test_select_energy_preserves_events_in_requested_interval(camera_events):
    selected = camera_events.select_energy(0.5 * u.TeV, 2 * u.TeV)

    # Energy intervals include the lower edge and exclude the upper edge.
    np.testing.assert_allclose(selected.energy.to_value(u.TeV), [0.8, 1.5])
    np.testing.assert_allclose(selected.x.to_value(u.deg), [-0.2, 0.2], atol=1e-6)
    np.testing.assert_allclose(selected.radius.to_value(u.deg), [0.2, 0.2], atol=1e-6)


def test_select_energy_accepts_unitless_tev_bounds(camera_events):
    selected = camera_events.select_energy(2, 10)

    np.testing.assert_allclose(selected.energy.to_value(u.TeV), [6, 3, 2])
