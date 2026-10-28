import astropy.units as u
import numpy as np
import pytest

from lst_tools import bkg
from lst_tools.bkg import ObservationFrame, SkyOffsetEvents


def make_events(frame, *, x=(-0.5, 0.25, 1.0), y=(0.0, 0.5, -0.25), energy=(0.2, 1.0, 4.0), livetime=10 * u.s):
    return SkyOffsetEvents(x=x, y=y, energy=energy, livetime=livetime, frame=frame)


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
