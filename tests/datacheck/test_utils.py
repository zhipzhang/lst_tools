import numpy as np
import pandas as pd
import pytest

from lst_tools.datacheck.utils import circular_mean_degrees, filter_paths_by_date, mean_fit_p_value_to_sigma


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


def test_filter_paths_by_date():
    files = [
        "DL1_datacheck_20220425.h5",
        "DL1_datacheck_20220426.h5",
        "DL1_datacheck_20220427.h5",
        "DL1_datacheck_20220428.h5",
    ]
    assert filter_paths_by_date(files, 20220426, 20220427) == [
        "DL1_datacheck_20220426.h5",
        "DL1_datacheck_20220427.h5",
    ]


def test_filter_paths_by_date_no_match():
    files = ["DL1_datacheck_20220426.h5"]
    assert filter_paths_by_date(files, 20230101, 20231231) == []


def test_filter_paths_by_date_ignores_date_in_directory():
    files = [
        "/data/20230101/DL1_datacheck_20220426.h5",
        "/data/20230101/DL1_datacheck_20230102.h5",
    ]
    assert filter_paths_by_date(files, 20230101, 20230131) == ["/data/20230101/DL1_datacheck_20230102.h5"]


def test_filter_paths_by_date_drops_files_without_date():
    files = ["DL1_datacheck_20220426.h5", "dl1_LST-1.Run10623.h5"]
    assert filter_paths_by_date(files, 20220101, 20230101) == ["DL1_datacheck_20220426.h5"]
