"""Behavioral contract for the center-free SkyOffsetEvents interface."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import lst_tools.bkg.events as events_module
from lst_tools import bkg
from lst_tools.bkg import SkyOffsetEvents


def make_events(
    *,
    x=(-0.5, 0.25, 1.0),
    y=(0.0, 0.5, -0.25),
    energy=(0.2, 1.0, 4.0),
    livetime=10 * u.s,
):
    return SkyOffsetEvents(x=x, y=y, energy=energy, livetime=livetime)


def test_events_are_center_free_and_use_canonical_units():
    events = SkyOffsetEvents(
        x=[-30, 15] * u.arcmin,
        y=[0, 30] * u.arcmin,
        energy=[200, 1_000] * u.GeV,
        livetime=2 * u.min,
    )

    assert not hasattr(events, "center")
    assert events.x.unit == u.deg
    assert events.y.unit == u.deg
    assert events.energy.unit == u.TeV
    assert events.livetime.unit == u.s
    np.testing.assert_allclose(events.x.value, [-0.5, 0.25])
    np.testing.assert_allclose(events.y.value, [0, 0.5])
    np.testing.assert_allclose(events.energy.value, [0.2, 1.0])
    assert events.livetime.value == pytest.approx(120)


def test_unitless_values_use_degrees_tev_and_seconds():
    events = make_events(livetime=10)

    assert events.x.unit == u.deg
    assert events.y.unit == u.deg
    assert events.energy.unit == u.TeV
    assert events.livetime == 10 * u.s


def test_empty_initializer_is_a_zero_exposure_sample():
    events = SkyOffsetEvents()

    assert len(events) == 0
    assert events.x.unit == u.deg
    assert events.y.unit == u.deg
    assert events.energy.unit == u.TeV
    assert events.livetime == 0 * u.s


def test_camera_events_has_been_removed():
    assert not hasattr(bkg, "CameraEvents")
    assert not hasattr(events_module, "CameraEvents")


def test_empty_initializer_is_an_additive_identity():
    events = make_events()

    combined = events + SkyOffsetEvents()

    np.testing.assert_allclose(combined.x, events.x)
    np.testing.assert_allclose(combined.y, events.y)
    np.testing.assert_allclose(combined.energy, events.energy)
    assert combined.livetime == events.livetime
    assert combined is not events


def test_input_arrays_are_copied():
    x = np.array([0.1, 0.2])
    events = SkyOffsetEvents(x=x, y=[0, 0], energy=[1, 2], livetime=3)

    x[0] = 99

    assert events.x[0] == 0.1 * u.deg


@pytest.mark.parametrize(
    ("x", "y", "energy"),
    [
        ([0, 1], [0], [1, 2]),
        ([[0, 1]], [[0, 1]], [[1, 2]]),
    ],
)
def test_event_columns_must_be_one_dimensional_and_equal_length(x, y, energy):
    with pytest.raises(ValueError, match="same number|one-dimensional"):
        SkyOffsetEvents(x=x, y=y, energy=energy, livetime=1)


@pytest.mark.parametrize("livetime", [-1, np.inf, [1, 2], 1 * u.m])
def test_livetime_must_be_a_finite_non_negative_scalar_time(livetime):
    with pytest.raises(ValueError, match="livetime"):
        SkyOffsetEvents(x=[], y=[], energy=[], livetime=livetime)


def test_empty_sample_can_represent_exposure_without_selected_events():
    events = SkyOffsetEvents(x=[], y=[], energy=[], livetime=30 * u.s)

    assert len(events) == 0
    assert events.livetime == 30 * u.s


def test_non_empty_sample_requires_positive_livetime():
    with pytest.raises(ValueError, match="positive"):
        SkyOffsetEvents(x=[0], y=[0], energy=[1], livetime=0 * u.s)


def test_radius_is_exact_spherical_separation():
    events = SkyOffsetEvents(x=[1], y=[1], energy=[1], livetime=1)
    offset_event = SkyCoord(ra=1 * u.deg, dec=1 * u.deg, frame="icrs")
    origin = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs")

    assert events.radius[0] == offset_event.separation(origin)


def test_energy_selection_is_half_open_and_preserves_full_livetime():
    events = make_events(energy=(0.2, 1.0, 2.0, 4.0), x=(0, 1, 2, 3), y=(0, 0, 0, 0))

    selected = events.select_energy(1 * u.TeV, 4 * u.TeV)

    np.testing.assert_allclose(selected.energy.value, [1, 2])
    np.testing.assert_allclose(selected.x.value, [1, 2])
    assert selected.livetime == events.livetime


def test_to_image_uses_requested_bins():
    events = SkyOffsetEvents(
        x=[-0.8, -0.2, 0.2, 0.8, 0, 2] * u.deg,
        y=[0, 0, 0, 0, 0, 0] * u.deg,
        energy=[0.2, 0.8, 1.5, 6, 3, 2] * u.TeV,
        livetime=10 * u.s,
    )

    image = events.to_image(
        x_edges=[-1, 0, 1] * u.deg,
        y_edges=[-0.5, 0.5] * u.deg,
        e_edges=[0.1, 1, 10] * u.TeV,
    )

    expected = np.zeros((2, 1, 2))
    expected[0, 0, 0] = 2
    expected[1, 0, 1] = 3
    np.testing.assert_array_equal(image.histogram.values(), expected)


def test_add_concatenates_events_and_sums_livetime():
    first = make_events(x=(0, 1), y=(0, 0), energy=(0.2, 1), livetime=10 * u.s)
    second = make_events(x=(2,), y=(0.5,), energy=(3,), livetime=0.5 * u.min)

    combined = first.add(second)

    np.testing.assert_allclose(combined.x.value, [0, 1, 2])
    np.testing.assert_allclose(combined.y.value, [0, 0, 0.5])
    np.testing.assert_allclose(combined.energy.value, [0.2, 1, 3])
    assert combined.livetime == 40 * u.s


def test_add_includes_livetime_from_an_empty_sample():
    events = make_events(livetime=10 * u.s)
    empty_exposure = SkyOffsetEvents(x=[], y=[], energy=[], livetime=20 * u.s)

    combined = events.add(empty_exposure)

    np.testing.assert_allclose(combined.energy, events.energy)
    assert combined.livetime == 30 * u.s


def test_add_does_not_modify_or_share_arrays_with_operands():
    first = make_events(x=(0,), y=(0,), energy=(1,), livetime=10)
    second = make_events(x=(1,), y=(1,), energy=(2,), livetime=20)

    combined = first.add(second)
    combined.x[0] = 99 * u.deg

    assert first.x[0] == 0 * u.deg
    assert second.x[0] == 1 * u.deg
    assert first.livetime == 10 * u.s
    assert second.livetime == 20 * u.s


def test_plus_operator_has_the_same_semantics_as_add():
    first = make_events(x=(0,), y=(0,), energy=(1,), livetime=10)
    second = make_events(x=(1,), y=(1,), energy=(2,), livetime=20)

    combined = first + second

    np.testing.assert_allclose(combined.energy.value, [1, 2])
    assert combined.livetime == 30 * u.s


def test_add_rejects_other_types():
    events = make_events()

    with pytest.raises(TypeError, match="SkyOffsetEvents"):
        events.add(object())
