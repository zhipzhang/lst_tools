import astropy.units as u
import numpy as np
import pytest
from gammapy.irf import Background2D

from lst_tools.bkg import Background2DMaker, SkyOffsetEvents


@pytest.fixture
def events():
    return SkyOffsetEvents(
        x=[0.5, 1.5, 0.5, 1.5, 3.0] * u.deg,
        y=np.zeros(5) * u.deg,
        energy=[0.2, 0.2, 2.0, 2.0, 2.0] * u.TeV,
        livetime=10 * u.s,
    )


@pytest.fixture
def maker():
    return Background2DMaker(
        energy_edges=[0.1, 1, 10] * u.TeV,
        theta_edges=[0, 1, 2] * u.deg,
    )


def expected_solid_angles():
    theta = np.deg2rad([0, 1, 2])
    return np.diff(2 * np.pi * (1 - np.cos(theta))) * u.sr


def effective_widths(energy_edges, spectral_index):
    low = energy_edges[:-1].to(u.MeV)
    high = energy_edges[1:].to(u.MeV)
    reference = np.sqrt(low * high)
    if spectral_index == -1:
        return reference * np.log(high / low)
    exponent = spectral_index + 1
    return reference * (
        (high / reference).to_value(u.one) ** exponent
        - (low / reference).to_value(u.one) ** exponent
    ) / exponent


def test_counts_use_energy_and_theta_bins(maker, events):
    counts = maker.counts(events)

    np.testing.assert_array_equal(counts, [[1, 1], [1, 1]])
    assert counts.dtype == np.int64


def test_events_outside_configured_axes_are_discarded(maker, events):
    assert maker.counts(events).sum() == 4


@pytest.mark.parametrize("spectral_index", [-2, -1, 0.5])
def test_rate_preserves_integral_counts(events, spectral_index):
    maker = Background2DMaker(
        energy_edges=[0.1, 1, 10] * u.TeV,
        theta_edges=[0, 1, 2] * u.deg,
        spectral_index=spectral_index,
    )
    widths = effective_widths(maker.energy_edges, spectral_index)
    rate = maker.differential_rate(events)
    recovered_counts = rate * events.livetime * widths[:, np.newaxis]
    recovered_counts *= expected_solid_angles()[np.newaxis, :]

    assert rate.unit == 1 / (u.MeV * u.s * u.sr)
    np.testing.assert_allclose(recovered_counts.to_value(u.one), maker.counts(events))


def test_call_returns_background_with_gammapy_axes_data_and_unit(maker, events):
    background = maker(events)
    rate = maker.differential_rate(events)

    assert isinstance(background, Background2D)
    assert background.axes.names == ["energy", "offset"]
    np.testing.assert_allclose(background.axes["energy"].edges, [0.1, 1, 10] * u.TeV)
    np.testing.assert_allclose(background.axes["offset"].edges, [0, 1, 2] * u.deg)
    np.testing.assert_allclose(background.data, rate.value)
    assert background.unit == rate.unit
    assert background.meta["LIVETIME"] == pytest.approx(10)
    assert background.meta["SPEC_IDX"] == pytest.approx(-2)


def test_empty_positive_exposure_produces_zero_background():
    maker = Background2DMaker(
        energy_edges=[0.1, 1, 10],
        theta_edges=[0, 1, 2],
    )
    events = SkyOffsetEvents(livetime=10 * u.s)

    np.testing.assert_array_equal(maker.counts(events), np.zeros((2, 2)))
    np.testing.assert_array_equal(maker.differential_rate(events).value, np.zeros((2, 2)))
    np.testing.assert_array_equal(maker(events).data, np.zeros((2, 2)))


def test_maker_initialization_only_normalizes_configuration():
    energy_edges = np.array([0.1, 1, 10])
    maker = Background2DMaker(energy_edges, [0, 1, 2])

    energy_edges[1] = 5

    assert not hasattr(maker, "events")
    np.testing.assert_allclose(maker.energy_edges.to_value(u.TeV), [0.1, 1, 10])


def test_maker_can_be_reused_for_different_event_samples(maker, events):
    populated = maker(events)
    empty = maker(SkyOffsetEvents(livetime=10 * u.s))

    assert populated.data.sum() > 0
    assert empty.data.sum() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energy_edges": [0, 1, 10]}, "positive"),
        ({"energy_edges": [0.1, 1, 1]}, "energy_edges"),
        ({"theta_edges": [-1, 0, 1]}, "between 0 and 180"),
        ({"theta_edges": [0, 90, 181]}, "between 0 and 180"),
        ({"spectral_index": np.inf}, "spectral_index"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    arguments = {
        "energy_edges": [0.1, 1, 10],
        "theta_edges": [0, 1, 2],
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        Background2DMaker(**arguments)


@pytest.mark.parametrize("events", [object(), SkyOffsetEvents()])
def test_call_rejects_invalid_events(maker, events):
    message = "SkyOffsetEvents" if not isinstance(events, SkyOffsetEvents) else "livetime"

    with pytest.raises((TypeError, ValueError), match=message):
        maker(events)


def test_returned_background_is_fits_writable(maker, events, tmp_path):
    filename = tmp_path / "background.fits"
    background = maker(events)

    background.write(filename)
    restored = Background2D.read(filename)

    np.testing.assert_allclose(restored.data, background.data)
    assert restored.unit == background.unit
    np.testing.assert_allclose(restored.axes["energy"].edges, background.axes["energy"].edges)
    np.testing.assert_allclose(restored.axes["offset"].edges, background.axes["offset"].edges)
