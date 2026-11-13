import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from lst_tools.datacheck import RunQualityFilter, plot_advanced_distributions


@pytest.fixture
def sample_statistics() -> dict:
    """Build basic-only and advanced-cut run statistics."""
    return {
        "empty_df": pd.DataFrame(),
        "lack_basic_columns": pd.DataFrame({"n_subruns": [1, 2, 3], "date": [20220101, 20220102, 20220103]}),
        "lack_advanced": pd.DataFrame(
            {
                "runnumber": [1, 2, 3],
                "n_subruns": [1, 2, 3],
                "date": [20220101, 20220102, 20220103],
                "n_flatfield": [10, 20, 30],
                "n_pedestal": [100, 200, 300],
                "mean_ra": [10.4, 10.4, 10.4],
                "mean_dec": [10, 20, 30],
                "mean_cos_zd": [0.1, 0.2, 0.3],
                "observation_time": [1800, 1800, 1800],
                "pointing_dec_std": [0.01, 0.02, 0.03],
            }
        ),
        "full_df": pd.DataFrame(
            {
                "runnumber": [1, 2, 3, 4, 5, 6, 7, 8],
                "n_subruns": [1, 1, 1, 4, 1, 1, 1, 1],
                "date": [20220101] * 8,
                "n_flatfield": [10] * 8,
                "n_pedestal": [100] * 8,
                "mean_ra": [10.4] * 8,
                "mean_dec": [10] * 8,
                "mean_cos_zd": [0.5] * 8,
                "observation_time": [1800] * 8,
                "pointing_dec_std": [0.01] * 8,
                # Run 1 passes every advanced cut. Runs 2–8 each fail one
                # configurable threshold in the order tested below.
                "mean_diffuse_nsb_std": [2.3, 2.4, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
                "mean_intensity_threshold": [50, 50, 51, 50, 50, 50, 50, 50],
                "mean_fit_p_value": [0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
                "mean_index": [-2.2, -2.2, -2.2, -2.2, -2.4, -2.0, -2.2, -2.2],
                "mean_R422": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.4, 1.5],
                "fraction_around_mode_R422": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7],
            }
        ),
    }


def make_basic_statistics(**overrides):
    values = {
        "runnumber": 1,
        "n_subruns": 1,
        "date": 20220101,
        "n_flatfield": 1,
        "n_pedestal": 1,
        "mean_ra": 10.0,
        "mean_dec": 10.0,
        "mean_cos_zd": np.cos(np.radians(20.0)),
        "observation_time": 1800.0,
        "pointing_dec_std": 0.01,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def make_basic_filter(**overrides):
    options = {
        "source_ra": 10.0,
        "source_dec": 10.0,
        "min_angle_to_source": 0.0,
        "max_angle_to_source": 30.0,
        "max_zenith_angle": 20.0,
        "min_observation_time": 1800.0,
    }
    options.update(overrides)
    return RunQualityFilter(**options)


def test_missing_basic_columns_are_reported(sample_statistics: dict):
    data_filter = RunQualityFilter()
    with pytest.raises(ValueError, match="runnumber"):
        data_filter.apply_basic_cuts(sample_statistics["empty_df"])
    with pytest.raises(ValueError, match="mean_ra"):
        data_filter.apply_basic_cuts(sample_statistics["lack_basic_columns"])


def test_missing_advanced_columns_are_reported(sample_statistics: dict):
    data_filter = RunQualityFilter()
    with pytest.raises(ValueError, match="mean_diffuse_nsb_std"):
        data_filter(sample_statistics["lack_advanced"], advanced_cuts=True)


def test_basic_cuts_accept_a_valid_run_at_inclusive_thresholds():
    result = make_basic_filter().apply_basic_cuts(make_basic_statistics())

    assert result["runnumber"].tolist() == [1]


@pytest.mark.parametrize(
    ("statistics_override", "filter_override"),
    [
        ({"n_subruns": 0}, {}),
        ({"date": 20230101}, {"last_date": 20221231}),
        ({"n_flatfield": 0}, {}),
        ({"n_pedestal": 0}, {}),
        ({"mean_ra": 11.0}, {"max_angle_to_source": 0.5}),
        ({"mean_cos_zd": np.cos(np.radians(30.0))}, {}),
        ({"observation_time": 1799.0}, {}),
        ({"pointing_dec_std": 0.02}, {}),
    ],
)
def test_basic_cuts_reject_each_failed_condition(statistics_override, filter_override):
    result = make_basic_filter(**filter_override).apply_basic_cuts(make_basic_statistics(**statistics_override))

    assert result.empty


def test_basic_cuts_use_absolute_galactic_latitude():
    result = make_basic_filter(min_galactic_lat=80.0).apply_basic_cuts(make_basic_statistics())

    assert result.empty


def test_basic_cuts_handle_right_ascension_wraparound():
    statistics = make_basic_statistics(mean_ra=0.1)
    data_filter = make_basic_filter(source_ra=359.9, max_angle_to_source=0.5)

    assert data_filter.apply_basic_cuts(statistics)["runnumber"].tolist() == [1]


@pytest.mark.parametrize("column", ["mean_ra", "observation_time", "pointing_dec_std"])
def test_basic_cuts_reject_nan_inputs(column):
    statistics = make_basic_statistics(**{column: np.nan})

    assert make_basic_filter().apply_basic_cuts(statistics).empty


def test_apply_basic_cuts(sample_statistics: dict):
    data_filter = RunQualityFilter(
        source_ra=10, source_dec=10, max_pointing_dec_std=0.015, min_angle_to_source=0, max_angle_to_source=30
    )
    result = data_filter.apply_basic_cuts(sample_statistics["lack_advanced"])
    assert isinstance(result, pd.DataFrame)
    assert result["runnumber"].tolist() == [1]


def test_apply_basic_cuts_min_observation_time(sample_statistics: dict):
    data_filter = RunQualityFilter(
        source_ra=10,
        source_dec=10,
        max_pointing_dec_std=0.015,
        min_angle_to_source=0,
        max_angle_to_source=30,
        min_observation_time=1801,
    )
    result = data_filter.apply_basic_cuts(sample_statistics["lack_advanced"])
    assert result.empty


def test_apply_advanced_cuts(sample_statistics: dict):
    data_filter = RunQualityFilter(source_ra=10, source_dec=10)
    result = data_filter.apply_advanced_cuts(sample_statistics["full_df"])
    assert isinstance(result, pd.DataFrame)
    assert result["runnumber"].tolist() == [1]
    assert data_filter(sample_statistics["full_df"], advanced_cuts=True) == [1]


def test_advanced_cuts_reject_nan_inputs(sample_statistics: dict):
    statistics = sample_statistics["full_df"].copy()
    statistics.loc[0, "mean_R422"] = np.nan

    result = RunQualityFilter().apply_advanced_cuts(statistics)

    assert result.empty


@pytest.mark.parametrize(
    ("threshold", "value", "newly_accepted_run"),
    [
        ("max_diffuse_nsb_std", 2.41, 2),
        ("max_intensity_at_half_peak_rate", 51.1, 3),
        ("min_mean_fit_p_sigma", -3.5, 4),
        ("min_drdi_index", -2.41, 5),
        ("max_drdi_index", -1.99, 6),
        ("min_drdi_at_422pe", 1.39, 7),
        ("min_fraction_around_mode", 0.69, 8),
    ],
)
def test_relax_each_advanced_cut(sample_statistics: dict, threshold: str, value: float, newly_accepted_run: int):
    data_filter = RunQualityFilter(source_ra=10, source_dec=10, **{threshold: value})
    result = data_filter.apply_advanced_cuts(sample_statistics["full_df"])
    assert result["runnumber"].tolist() == [1, newly_accepted_run]


def test_plot_advanced_distributions_exposes_unfiltered_figures(sample_statistics: dict):
    data_filter = RunQualityFilter(source_ra=10, source_dec=10)
    statistics = sample_statistics["full_df"].copy()
    statistics.loc[0, "n_subruns"] = 0  # Would be removed by the basic cuts.

    figures = plot_advanced_distributions(statistics, data_filter)

    assert set(figures) == set(RunQualityFilter.ADVANCED_COLUMNS)
    assert all(isinstance(figure, Figure) for figure in figures.values())
    assert all(len(figure.axes) == 1 for figure in figures.values())
    assert all(
        sum(patch.get_height() for patch in figure.axes[0].patches) == len(statistics) for figure in figures.values()
    )

    expected = {
        "mean_diffuse_nsb_std": ([2.3], 7),
        "mean_intensity_threshold": ([50.0], 7),
        "mean_fit_p_value": ([-3.0], 7),
        "mean_index": ([-2.35, -2.1], 6),
        "mean_R422": ([1.5], 7),
        "fraction_around_mode_R422": ([0.8], 7),
    }
    for name, (thresholds, n_passed) in expected.items():
        axis = figures[name].axes[0]
        assert [line.get_xdata()[0] for line in axis.lines] == thresholds
        assert axis.texts[0].get_text() == f"Pass: {n_passed}/8\nFail: {8 - n_passed}/8"

    for figure in figures.values():
        plt.close(figure)
