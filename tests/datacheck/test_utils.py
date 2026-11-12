import numpy as np
import pandas as pd
import pytest

from lst_tools.datacheck.utils import circular_mean_degrees, mean_fit_p_value_to_sigma


def test_mean_fit_p_value_to_sigma():
    mean_p_value = pd.Series([0.5, 0.25, 0.0])
    n_subruns = pd.Series([3, 3, 12])

    result = mean_fit_p_value_to_sigma(mean_p_value, n_subruns)

    np.testing.assert_allclose(result, [0.0, -1.5, -6.0])


# TEST: sliding_window_mode needs a dedicated scientific review of its cumulative-window
# indexing before correct reference results can be pinned. Its current algorithm is unchanged.
@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ([0.015, 359.99], 0.0025),
        ([10.0, 20.0], 15.0),
    ],
)
def test_circular_mean_degrees(angles, expected):
    assert circular_mean_degrees(angles) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("angles", [[], [np.nan, np.nan]])
def test_circular_mean_degrees_returns_nan_without_finite_angles(angles):
    assert np.isnan(circular_mean_degrees(angles))
