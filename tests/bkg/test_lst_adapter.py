from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import AltAz, SkyCoord, SkyOffsetFrame
from astropy.time import Time
from astropy.utils import iers
from gammapy.utils.coordinates import FoVAltAzFrame

from lst_tools.bkg import SkyOffsetEvents, sky_offset_events_from_lstdl2
from lst_tools.location import LST_LOCATION


@pytest.fixture(scope="module", autouse=True)
def use_bundled_iers_data():
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.fixture
def run_dl2():
    """DL2-like data for one run with a fixed telescope pointing."""
    pointing = SkyCoord(ra=83.5, dec=20.3, unit=u.deg, frame="icrs")
    event_time = Time(1_700_000_000 + np.arange(4) * 60, format="unix")
    expected_x = np.array([-0.8, -0.2, 0.3, 0.9])
    expected_y = np.array([0.1, -0.3, 0.4, -0.2])
    reconstructed = SkyCoord(
        lon=expected_x * u.deg,
        lat=expected_y * u.deg,
        frame=SkyOffsetFrame(origin=pointing),
    ).icrs
    altaz_frame = AltAz(obstime=event_time, location=LST_LOCATION)
    reconstructed_altaz = reconstructed.transform_to(altaz_frame)
    data = pd.DataFrame(
        {
            "reco_alt": reconstructed_altaz.alt.to_value(u.rad),
            "reco_az": reconstructed_altaz.az.to_value(u.rad),
            "reco_energy": [0.2, 0.8, 2.0, 5.0],
            "trigger_time": event_time.unix,
            "keep": [True, False, True, False],
        }
    )
    dl2 = SimpleNamespace(
        data=data,
        t_eff=12.5,
        pointing_ra=pointing.ra,
        pointing_dec=pointing.dec,
    )
    return dl2, expected_x, expected_y


def test_adapter_uses_run_pointing_as_frame_center(run_dl2):
    dl2, expected_x, expected_y = run_dl2

    events = sky_offset_events_from_lstdl2(dl2)

    assert isinstance(events, SkyOffsetEvents)
    assert not hasattr(events, "center")
    # FoV frame longitude increases opposite to SkyOffsetFrame longitude
    np.testing.assert_allclose(events.x, -expected_x, atol=1e-8)
    np.testing.assert_allclose(events.y, expected_y, atol=1e-8)
    np.testing.assert_allclose(events.energy, [0.2, 0.8, 2.0, 5.0])
    assert events.livetime == 12.5 * u.s


def test_adapter_applies_filter_and_preserves_complete_livetime(run_dl2):
    dl2, _, _ = run_dl2

    events = sky_offset_events_from_lstdl2(dl2, event_filter=lambda data: data.loc[data["keep"]])

    assert len(events) == 2
    np.testing.assert_allclose(events.energy, [0.2, 2.0])
    assert events.livetime == 12.5 * u.s


def test_adapter_returns_empty_exposure_when_no_events_survive(run_dl2):
    dl2, _, _ = run_dl2

    events = sky_offset_events_from_lstdl2(dl2, event_filter=lambda data: data.iloc[:0])

    assert len(events) == 0
    assert events.livetime == 12.5 * u.s


def test_adapter_accepts_quantity_livetime(run_dl2):
    dl2, _, _ = run_dl2
    dl2.t_eff = 2 * u.min

    events = sky_offset_events_from_lstdl2(dl2)

    assert events.livetime == 120 * u.s


def test_adapter_does_not_modify_input_dataframe(run_dl2):
    dl2, _, _ = run_dl2
    original = dl2.data.copy(deep=True)

    sky_offset_events_from_lstdl2(dl2)

    pd.testing.assert_frame_equal(dl2.data, original)


def test_adapter_requires_data_pointing_and_effective_livetime():
    with pytest.raises(TypeError, match="data.*t_eff"):
        sky_offset_events_from_lstdl2(object())


def test_adapter_requires_dataframe_data(run_dl2):
    dl2, _, _ = run_dl2
    dl2.data = []

    with pytest.raises(TypeError, match="DataFrame"):
        sky_offset_events_from_lstdl2(dl2)


def test_adapter_validates_effective_livetime(run_dl2):
    dl2, _, _ = run_dl2
    dl2.t_eff = -1

    with pytest.raises(ValueError, match="livetime"):
        sky_offset_events_from_lstdl2(dl2)


@pytest.fixture
def run_dl2_altaz():
    """DL2-like data with 3 events, each with its own pointing and time."""
    event_time = Time(1_700_000_000 + np.array([0.0, 60.0, 120.0]), format="unix")
    pointing_alt = np.deg2rad([70.0, 65.0, 60.0])
    pointing_az = np.deg2rad([10.0, 180.0, 350.0])
    data = pd.DataFrame(
        {
            "reco_alt": pointing_alt + np.deg2rad([0.1, -0.2, 0.3]),
            "reco_az": pointing_az + np.deg2rad([0.2, 0.1, -0.15]),
            "pointing_alt": pointing_alt,
            "pointing_az": pointing_az,
            "reco_energy": [0.2, 0.8, 2.0],
            "trigger_time": event_time.unix,
        }
    )
    dl2 = SimpleNamespace(data=data, t_eff=30.0)
    return dl2, event_time


def test_adapter_altaz_frame_matches_scalar_per_event_reference(run_dl2_altaz):
    dl2, event_time = run_dl2_altaz

    events = sky_offset_events_from_lstdl2(dl2, frame="altaz")

    expected_x, expected_y = [], []
    for row, time in zip(dl2.data.itertuples(), event_time):
        altaz_frame = AltAz(obstime=time, location=LST_LOCATION)
        origin = SkyCoord(alt=row.pointing_alt * u.rad, az=row.pointing_az * u.rad, frame=altaz_frame)
        direction = SkyCoord(alt=row.reco_alt * u.rad, az=row.reco_az * u.rad, frame=altaz_frame)
        fov = direction.transform_to(FoVAltAzFrame(origin=origin, location=LST_LOCATION))
        expected_x.append(fov.fov_lon.to_value(u.deg))
        expected_y.append(fov.fov_lat.to_value(u.deg))

    np.testing.assert_allclose(events.x, expected_x, atol=1e-10)
    np.testing.assert_allclose(events.y, expected_y, atol=1e-10)
    np.testing.assert_allclose(events.energy, [0.2, 0.8, 2.0])
    assert events.livetime == 30 * u.s


def test_adapter_altaz_frame_centers_each_event_on_its_own_pointing(run_dl2_altaz):
    dl2, _ = run_dl2_altaz
    dl2.data["reco_alt"] = dl2.data["pointing_alt"]
    dl2.data["reco_az"] = dl2.data["pointing_az"]

    events = sky_offset_events_from_lstdl2(dl2, frame="altaz")

    np.testing.assert_allclose(events.x, 0, atol=1e-10)
    np.testing.assert_allclose(events.y, 0, atol=1e-10)
