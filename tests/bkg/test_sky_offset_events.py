import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from gammapy.utils.coordinates import FoVAltAzFrame

from lst_tools import bkg
from lst_tools.bkg import ObservationFrame, SkyOffsetEvents
from lst_tools.location import LST_LOCATION


def make_events(frame, *, x=(-0.5, 0.25, 1.0), y=(0.0, 0.5, -0.25), energy=(0.2, 1.0, 4.0), livetime=10 * u.s):
    return SkyOffsetEvents(x=x, y=y, energy=energy, livetime=livetime, frame=frame)


def make_altaz_events(offsets=(0, 600, 1200), *, livetime=120 * u.s):
    times = Time(1_700_000_000 + np.asarray(offsets), format="unix")
    pointing = SkyCoord(ra=83.5, dec=20.3, unit=u.deg, frame="icrs")
    origin = pointing.transform_to(AltAz(obstime=times, location=LST_LOCATION))
    mean_pointing = pointing.transform_to(AltAz(obstime=times.mean(), location=LST_LOCATION))
    if len(times) == 1:
        origin = origin[0]
    frame = ObservationFrame(
        fov_frame=FoVAltAzFrame(origin=origin, location=LST_LOCATION),
        mean_time=times.mean(),
        pointing_alt=mean_pointing.alt,
        pointing_az=mean_pointing.az,
    )
    return SkyOffsetEvents(
        x=np.linspace(0, 0.2, len(times)),
        y=np.zeros(len(times)),
        energy=np.arange(len(times)) + 1,
        livetime=livetime,
        frame=frame,
    )


def test_events_use_canonical_units_and_keep_frame(observation_frame):
    events = SkyOffsetEvents(
        x=[-30, 15] * u.arcmin,
        y=[0, 30] * u.arcmin,
        energy=[200, 1_000] * u.GeV,
        livetime=2 * u.min,
        frame=observation_frame,
    )

    assert events.frame is observation_frame
    assert events.livetime.unit == u.s
    np.testing.assert_allclose(events.x, [-0.5, 0.25])
    np.testing.assert_allclose(events.y, [0, 0.5])
    np.testing.assert_allclose(events.energy, [0.2, 1.0])
    assert events.livetime.value == pytest.approx(120)


def test_frame_is_required_even_for_empty_samples():
    with pytest.raises(TypeError):
        SkyOffsetEvents()


def test_empty_initializer_is_a_zero_exposure_sample(observation_frame):
    events = SkyOffsetEvents(frame=observation_frame)

    assert len(events) == 0
    assert events.livetime == 0 * u.s


def test_observation_frame_contains_fov_and_mean_pointing(observation_frame):
    assert observation_frame.mean_time.isscalar
    assert observation_frame.pointing_alt == 70 * u.deg
    assert observation_frame.pointing_az == 10 * u.deg


def test_event_addition_and_exclusion_state_are_not_supported(observation_frame):
    events = make_events(observation_frame)

    assert not hasattr(events, "add")
    assert not hasattr(events, "exclusion_regions")
    with pytest.raises(TypeError):
        events + events


def test_camera_events_has_been_removed():
    assert not hasattr(bkg, "CameraEvents")


def test_input_arrays_are_copied(observation_frame):
    x = np.array([0.1, 0.2])
    events = SkyOffsetEvents(x=x, y=[0, 0], energy=[1, 2], livetime=3 * u.s, frame=observation_frame)

    x[0] = 99

    assert events.x[0] == 0.1


@pytest.mark.parametrize(
    ("x", "y", "energy"),
    [([0, 1], [0], [1, 2]), ([[0, 1]], [[0, 1]], [[1, 2]])],
)
def test_event_columns_must_be_one_dimensional_and_equal_length(observation_frame, x, y, energy):
    with pytest.raises(ValueError, match="same number|one-dimensional"):
        SkyOffsetEvents(x=x, y=y, energy=energy, livetime=1 * u.s, frame=observation_frame)


@pytest.mark.parametrize("livetime", [-1 * u.s, np.inf * u.s, [1, 2] * u.s, 1 * u.m])
def test_livetime_must_be_a_finite_non_negative_scalar_time(observation_frame, livetime):
    with pytest.raises(ValueError, match="livetime"):
        SkyOffsetEvents(livetime=livetime, frame=observation_frame)


def test_empty_sample_can_represent_exposure_without_selected_events(observation_frame):
    events = SkyOffsetEvents(livetime=30 * u.s, frame=observation_frame)

    assert len(events) == 0
    assert events.livetime == 30 * u.s


def test_non_empty_sample_requires_positive_livetime(observation_frame):
    with pytest.raises(ValueError, match="positive"):
        SkyOffsetEvents(x=[0], y=[0], energy=[1], livetime=0 * u.s, frame=observation_frame)


def test_radius_is_the_flat_offset_distance(observation_frame):
    events = make_events(observation_frame, x=[1, 3], y=[1, 4], energy=[1, 1], livetime=1 * u.s)

    np.testing.assert_allclose(events.radius, [np.sqrt(2), 5])


def test_energy_selection_preserves_livetime_and_frame(observation_frame):
    events = make_events(observation_frame, energy=(0.2, 1.0, 2.0, 4.0), x=(0, 1, 2, 3), y=(0, 0, 0, 0))

    selected = events.select_energy(1 * u.TeV, 4 * u.TeV)

    np.testing.assert_allclose(selected.energy, [1, 2])
    np.testing.assert_allclose(selected.x, [1, 2])
    assert selected.livetime == events.livetime
    assert selected.frame is observation_frame


def test_time_slices_require_altaz_frame(observation_frame):
    with pytest.raises(TypeError, match="FoVAltAzFrame"):
        make_events(observation_frame).time_slices()


def test_time_slices_conserve_events_and_livetime():
    events = make_altaz_events()

    slices = events.time_slices()

    assert len(slices) > 1
    assert sum(map(len, slices)) == len(events)
    assert sum((slice_.livetime for slice_ in slices), 0 * u.s) == events.livetime
    np.testing.assert_allclose(np.concatenate([slice_.energy for slice_ in slices]), events.energy)
    assert all(slice_.frame.fov_frame.origin.isscalar for slice_ in slices)
    assert all(slice_.frame.fov_frame.obstime.isscalar for slice_ in slices)
    assert any(len(slice_) == 0 for slice_ in slices)


def test_time_slices_have_a_one_second_minimum_duration():
    events = make_altaz_events((0, 1, 2), livetime=10 * u.s)

    slices = events.time_slices(max_fov_rotation=1e-12 * u.deg)

    assert len(slices) == 2
    assert [len(slice_) for slice_ in slices] == [1, 2]
    assert all(slice_.livetime == 5 * u.s for slice_ in slices)


def test_time_slices_return_fixed_frame_events_unchanged():
    events = make_altaz_events((0,))

    slices = events.time_slices()

    assert slices == [events]


def test_time_slices_handle_east_west_pointing_singularity():
    # High-elevation run centered exactly on due East (az = 90 deg at midpoint).
    mid_time = Time(1_700_000_000, format="unix")
    pointing = SkyCoord(alt=82 * u.deg, az=90 * u.deg, frame=AltAz(obstime=mid_time, location=LST_LOCATION)).icrs
    times = mid_time + np.array([-600.0, 0.0, 600.0]) * u.s
    origin = pointing.transform_to(AltAz(obstime=times, location=LST_LOCATION))
    mid_pointing = pointing.transform_to(AltAz(obstime=mid_time, location=LST_LOCATION))
    frame = ObservationFrame(
        fov_frame=FoVAltAzFrame(origin=origin, location=LST_LOCATION),
        mean_time=mid_time,
        pointing_alt=mid_pointing.alt,
        pointing_az=90 * u.deg,
    )
    events = SkyOffsetEvents(x=[0, 0, 0], y=[0, 0, 0], energy=[1, 2, 3], livetime=1200 * u.s, frame=frame)

    slices = events.time_slices(max_fov_rotation=0.2 * u.deg)

    assert len(slices) > 1
    assert sum(map(len, slices)) == 3
    assert sum((slice_.livetime for slice_ in slices), 0 * u.s).to_value(u.s) == pytest.approx(1200.0)


def test_time_slices_use_mean_icrs_pointing_rather_than_first_event():
    times = Time(1_700_000_000 + np.array([0.0, 100.0, 200.0, 300.0, 400.0]), format="unix")
    true_pointing = SkyCoord(ra=83.5, dec=20.3, unit=u.deg, frame="icrs")
    jittered_first = SkyCoord(ra=84.5, dec=20.3, unit=u.deg, frame="icrs")
    icrs_per_event = SkyCoord(
        ra=[jittered_first.ra.deg, *([true_pointing.ra.deg] * 4)] * u.deg,
        dec=[jittered_first.dec.deg, *([true_pointing.dec.deg] * 4)] * u.deg,
        frame="icrs",
    )
    origin = icrs_per_event.transform_to(AltAz(obstime=times, location=LST_LOCATION))
    mean_pointing = true_pointing.transform_to(AltAz(obstime=times.mean(), location=LST_LOCATION))
    frame = ObservationFrame(
        fov_frame=FoVAltAzFrame(origin=origin, location=LST_LOCATION),
        mean_time=times.mean(),
        pointing_alt=mean_pointing.alt,
        pointing_az=mean_pointing.az,
    )
    events = SkyOffsetEvents(x=np.zeros(5), y=np.zeros(5), energy=np.ones(5), livetime=400 * u.s, frame=frame)

    slices = events.time_slices(0.5)
    slice_icrs = SkyCoord(slices[0].frame.fov_frame.origin).icrs

    assert slice_icrs.separation(true_pointing) < slice_icrs.separation(jittered_first)


def test_time_slices_preserve_empty_filtered_altaz_exposure():
    empty_origin = SkyCoord(
        alt=np.array([]) * u.deg,
        az=np.array([]) * u.deg,
        frame=AltAz(obstime=Time(np.array([]), format="unix"), location=LST_LOCATION),
    )
    mean_time = Time(1_700_000_000, format="unix")
    frame = ObservationFrame(
        fov_frame=FoVAltAzFrame(origin=empty_origin, location=LST_LOCATION),
        mean_time=mean_time,
        pointing_alt=70 * u.deg,
        pointing_az=10 * u.deg,
    )
    events = SkyOffsetEvents(livetime=60 * u.s, frame=frame)

    slices = events.time_slices()

    assert len(slices) == 1
    assert len(slices[0]) == 0
    assert slices[0].livetime == 60 * u.s
    assert slices[0].frame.fov_frame.origin.isscalar
