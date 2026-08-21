import numpy as np
import pytest
from hist import Hist, axis, storage

from lst_ulities.dl4.plot import (
    _containment_68,
    _excess_histogram,
    _interpolate_containment,
    plot_excess_containment_68,
    plot_excess_counts,
)


ENERGY_EDGES = [0.1, 1.0, 10.0]
THETA_SQUARED_EDGES = [0.0, 0.01, 0.04, 0.09]


def make_histogram(values, variances=None, histogram_storage=None):
    values = np.asarray(values, dtype=float)
    if variances is None:
        variances = values
    if histogram_storage is None:
        histogram_storage = storage.Weight()

    histogram = Hist(
        axis.Variable(
            ENERGY_EDGES,
            name="estimated_energy",
            label="Estimated energy [TeV]",
        ),
        axis.Variable(
            THETA_SQUARED_EDGES,
            name="theta_squared",
            label=r"$\theta^2$ [deg$^2$]",
        ),
        storage=histogram_storage,
    )
    if isinstance(histogram_storage, storage.Weight):
        histogram[...] = np.stack((values, variances), axis=-1)
    else:
        histogram[...] = values
    return histogram


def make_theta_histogram(values, variances=None):
    values = np.asarray(values, dtype=float)
    if variances is None:
        variances = values

    histogram = Hist(
        axis.Variable(THETA_SQUARED_EDGES, name="theta_squared"),
        storage=storage.Weight(),
    )
    histogram[...] = np.stack((values, variances), axis=-1)
    return histogram


class RecordingAxes:
    def __init__(self):
        self.errorbar_calls = []
        self.xscale = None
        self.xlabel = None
        self.ylabel = None

    def errorbar(self, x, y, yerr, **kwargs):
        self.errorbar_calls.append((np.asarray(x), np.asarray(y), np.asarray(yerr), kwargs))

    def set_xscale(self, scale):
        self.xscale = scale

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label


def test_excess_histogram_propagates_weight_variance():
    h_on = make_histogram([[10, 20, 30], [5, 7, 9]])
    h_off = make_histogram([[2, 4, 6], [1, 3, 5]])

    excess = _excess_histogram(h_on, h_off, alpha=0.5)

    assert np.allclose(
        excess.values(),
        [[9, 18, 27], [4.5, 5.5, 6.5]],
    )
    assert np.allclose(
        excess.variances(),
        [[10.5, 21, 31.5], [5.25, 7.75, 10.25]],
    )
    # The helper must not mutate either input histogram.
    assert np.allclose(h_on.values(), [[10, 20, 30], [5, 7, 9]])
    assert np.allclose(h_off.values(), [[2, 4, 6], [1, 3, 5]])


def test_excess_histogram_rejects_storage_without_variance():
    h_on = make_histogram([[1, 2, 3], [4, 5, 6]], histogram_storage=storage.Double())
    h_off = make_histogram([[0, 0, 0], [0, 0, 0]], histogram_storage=storage.Double())

    with pytest.raises(ValueError, match="track variances"):
        _excess_histogram(h_on, h_off, alpha=1.0)


def test_interpolate_containment_interpolates_in_theta():
    excess = make_theta_histogram([50, 50, 0])

    theta, index, bin_fraction, total = _interpolate_containment(excess, 0.68)

    assert total == pytest.approx(100)
    assert index == 1
    assert bin_fraction == pytest.approx(0.36)
    assert theta == pytest.approx(0.136)


def test_interpolate_containment_rejects_non_positive_excess():
    excess = make_theta_histogram([-2, 1, 0], variances=[2, 1, 0])

    theta, index, bin_fraction, total = _interpolate_containment(excess, 0.68)

    assert np.isnan(theta)
    assert index == -1
    assert np.isnan(bin_fraction)
    assert total == pytest.approx(-1)


def test_containment_68_returns_asymmetric_one_sigma_errors():
    h_on = make_histogram([[68, 32, 0], [0, 0, 0]])
    h_off = make_histogram([[0, 0, 0], [1, 1, 1]])
    excess = _excess_histogram(h_on, h_off, alpha=1.0)

    containment, error_low, error_high = _containment_68(excess)

    assert containment[0] == pytest.approx(0.1)
    assert error_low[0] > 0
    assert error_high[0] > error_low[0]
    assert np.isnan(containment[1])
    assert np.isnan(error_low[1])
    assert np.isnan(error_high[1])


def test_plot_excess_counts_uses_hist_projection_and_errors():
    h_on = make_histogram([[10, 20, 30], [5, 7, 9]])
    h_off = make_histogram([[2, 4, 6], [1, 3, 5]])
    axes = RecordingAxes()

    result = plot_excess_counts(h_on, h_off, ax=axes, alpha=0.5)

    assert result is axes
    x, excess, error, kwargs = axes.errorbar_calls[0]
    assert np.allclose(x, [0.55, 5.5])
    assert np.allclose(excess, [54, 16.5])
    assert np.allclose(error, np.sqrt([63, 23.25]))
    assert kwargs["fmt"] == "o"
    assert axes.xscale == "log"
    assert axes.xlabel == "Estimated energy [TeV]"
    assert axes.ylabel == "Excess counts"


def test_plot_containment_omits_energy_bins_without_positive_excess():
    h_on = make_histogram([[68, 32, 0], [0, 0, 0]])
    h_off = make_histogram([[0, 0, 0], [1, 1, 1]])
    axes = RecordingAxes()

    result = plot_excess_containment_68(h_on, h_off, ax=axes)

    assert result is axes
    energy, containment, errors, _ = axes.errorbar_calls[0]
    assert np.allclose(energy, [0.55])
    assert containment == pytest.approx([0.1])
    assert errors.shape == (2, 1)
    assert np.all(errors > 0)
