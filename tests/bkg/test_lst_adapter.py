from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import AltAz, SkyCoord, SkyOffsetFrame
from astropy.time import Time
from astropy.utils import iers

from lst_tools.bkg import SkyOffsetEvents, sky_offset_events_from_lstdl2
from lst_tools.bkg.lst_adapter import LST_LOCATION, REQUIRED_COLUMNS


@pytest.fixture(scope="module", autouse=True)
def use_bundled_iers_data():
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.fixture
def moving_pointing_dl2():
    """DL2-like data with a deliberately changing telescope pointing."""
    event_time = Time(1_700_000_000 + np.arange(4) * 60, format="unix")
    pointing = SkyCoord(
        ra=[82.0, 83.0, 84.0, 85.0] * u.deg,
        dec=[20.0, 20.2, 20.4, 20.6] * u.deg,
        frame="icrs",
    )
    expected_x = np.array([-0.8, -0.2, 0.3, 0.9]) * u.deg
    expected_y = np.array([0.1, -0.3, 0.4, -0.2]) * u.deg
    reconstructed = SkyCoord(
        lon=expected_x,
        lat=expected_y,
        frame=SkyOffsetFrame(origin=pointing),
    ).icrs
    altaz_frame = AltAz(obstime=event_time, location=LST_LOCATION)
    pointing_altaz = pointing.transform_to(altaz_frame)
    reconstructed_altaz = reconstructed.transform_to(altaz_frame)
    data = pd.DataFrame(
        {
            "reco_alt": reconstructed_altaz.alt.to_value(u.rad),
            "reco_az": reconstructed_altaz.az.to_value(u.rad),
            "reco_energy": [0.2, 0.8, 2.0, 5.0],
            "alt_tel": pointing_altaz.alt.to_value(u.rad),
            "az_tel": pointing_altaz.az.to_value(u.rad),
            "trigger_time": event_time.unix,
            "keep": [True, False, True, False],
        }
    )
    dl2 = SimpleNamespace(data=data, t_eff=12.5)
    return dl2, pointing, reconstructed


def test_adapter_uses_one_frame_centered_on_mean_pointing(moving_pointing_dl2):
    dl2, pointing, reconstructed = moving_pointing_dl2
    mean_cartesian = pointing.cartesian.xyz.mean(axis=1)
    mean_pointing = SkyCoord(
        x=mean_cartesian[0],
        y=mean_cartesian[1],
        z=mean_cartesian[2],
        representation_type="cartesian",
        frame="icrs",
    )
    expected = reconstructed.transform_to(SkyOffsetFrame(origin=mean_pointing))

    events = sky_offset_events_from_lstdl2(dl2)

    assert isinstance(events, SkyOffsetEvents)
    assert not hasattr(events, "center")
    np.testing.assert_allclose(events.x, expected.lon, atol=1e-8 * u.deg)
    np.testing.assert_allclose(events.y, expected.lat, atol=1e-8 * u.deg)
    np.testing.assert_allclose(events.energy.to_value(u.TeV), [0.2, 0.8, 2.0, 5.0])
    assert events.livetime == 12.5 * u.s


def test_adapter_applies_filter_and_preserves_complete_livetime(moving_pointing_dl2):
    dl2, _, _ = moving_pointing_dl2

    events = sky_offset_events_from_lstdl2(dl2, event_filter=lambda data: data.loc[data["keep"]])
    selected_dl2 = SimpleNamespace(data=dl2.data.loc[dl2.data["keep"]], t_eff=dl2.t_eff)
    expected = sky_offset_events_from_lstdl2(selected_dl2)

    np.testing.assert_allclose(events.x, expected.x, atol=1e-8 * u.deg)
    np.testing.assert_allclose(events.y, expected.y, atol=1e-8 * u.deg)
    np.testing.assert_allclose(events.energy.to_value(u.TeV), [0.2, 2.0])
    assert events.livetime == 12.5 * u.s


def test_adapter_returns_empty_exposure_when_no_events_survive(moving_pointing_dl2):
    dl2, _, _ = moving_pointing_dl2

    events = sky_offset_events_from_lstdl2(dl2, event_filter=lambda data: data.iloc[:0])

    assert len(events) == 0
    assert events.livetime == 12.5 * u.s


def test_adapter_does_not_modify_input_dataframe(moving_pointing_dl2):
    dl2, _, _ = moving_pointing_dl2
    original = dl2.data.copy(deep=True)

    sky_offset_events_from_lstdl2(dl2)

    pd.testing.assert_frame_equal(dl2.data, original)


@pytest.mark.parametrize("missing_column", REQUIRED_COLUMNS)
def test_adapter_requires_dl2_columns(moving_pointing_dl2, missing_column):
    dl2, _, _ = moving_pointing_dl2
    incomplete = SimpleNamespace(data=dl2.data.drop(columns=missing_column), t_eff=dl2.t_eff)

    with pytest.raises(ValueError, match=missing_column):
        sky_offset_events_from_lstdl2(incomplete)


def test_adapter_requires_data_and_effective_livetime():
    with pytest.raises(TypeError, match="data.*t_eff"):
        sky_offset_events_from_lstdl2(object())


def test_adapter_requires_dataframe_data():
    with pytest.raises(TypeError, match="DataFrame"):
        sky_offset_events_from_lstdl2(SimpleNamespace(data=[], t_eff=1))


def test_adapter_requires_filter_to_return_dataframe(moving_pointing_dl2):
    dl2, _, _ = moving_pointing_dl2

    with pytest.raises(TypeError, match="event_filter.*DataFrame"):
        sky_offset_events_from_lstdl2(dl2, event_filter=lambda data: np.ones(len(data), dtype=bool))


def test_adapter_validates_effective_livetime(moving_pointing_dl2):
    dl2, _, _ = moving_pointing_dl2
    invalid_livetime = SimpleNamespace(data=dl2.data, t_eff=-1)

    with pytest.raises(ValueError, match="livetime"):
        sky_offset_events_from_lstdl2(invalid_livetime)
