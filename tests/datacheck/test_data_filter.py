import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from lst_tools.datacheck import DataFilter, plot_advanced_distributions


@pytest.fixture
def sample_pd() -> dict:
    """Build a dict of pd.DataFrame"""
    return {
        "empty_df": pd.DataFrame(),
        "lack_basic_columns": pd.DataFrame({"n_subruns": [1, 2, 3], "date": [20220101, 20220102, 20220103]}),
        "lack_advanced": pd.DataFrame(
            {
                "run_number": [1, 2, 3],
                "n_subruns": [1, 2, 3],
                "date": [20220101, 20220102, 20220103],
                "n_flatfield": [10, 20, 30],
                "n_pedestal": [100, 200, 300],
                "mean_ra": [10.4, 10.4, 10.4],
                "mean_dec": [10, 20, 30],
                "mean_cos_zd": [0.1, 0.2, 0.3],
                "pointing_dec_std": [0.01, 0.02, 0.03],
            }
        ),
        "full_df": pd.DataFrame(
            {
                "run_number": [1, 2, 3, 4, 5, 6, 7, 8],
                "n_subruns": [1, 1, 1, 4, 1, 1, 1, 1],
                "date": [20220101] * 8,
                "n_flatfield": [10] * 8,
                "n_pedestal": [100] * 8,
                "mean_ra": [10.4] * 8,
                "mean_dec": [10] * 8,
                "mean_cos_zd": [0.5] * 8,
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


def test_lack_basic_columns(sample_pd: dict):
    data_filter = DataFilter()
    with pytest.raises(ValueError):
        data_filter.apply_basic_cuts(sample_pd["empty_df"])
    with pytest.raises(ValueError):
        data_filter.apply_basic_cuts(sample_pd["lack_basic_columns"])

    result = data_filter(sample_pd["lack_advanced"])
    assert isinstance(result, list)
    assert all(isinstance(item, int) for item in result)


def test_lack_advanced(sample_pd: dict):
    data_filter = DataFilter()
    with pytest.raises(ValueError):
        data_filter(sample_pd["lack_advanced"], advanced_cuts=True)


def test_apply_basic_cuts(sample_pd: dict):
    data_filter = DataFilter(
        source_ra=10, source_dec=10, max_pointing_dec_std=0.015, min_angle_to_source=0, max_angle_to_source=30
    )
    result = data_filter.apply_basic_cuts(sample_pd["lack_advanced"])
    assert isinstance(result, pd.DataFrame)
    assert result["run_number"].tolist() == [1]


def test_apply_advanced_cuts(sample_pd: dict):
    data_filter = DataFilter(source_ra=10, source_dec=10)
    result = data_filter.apply_advanced_cuts(sample_pd["full_df"])
    assert isinstance(result, pd.DataFrame)
    assert result["run_number"].tolist() == [1]
    assert data_filter(sample_pd["full_df"], advanced_cuts=True) == [1]


@pytest.mark.parametrize(
    ("threshold", "value", "newly_accepted_run"),
    [
        ("max_diffuse_nsb_std", 2.41, 2),
        ("max_intensity_at_half_peak_rate", 51.1, 3),
        ("min_mean_fit_p", -3.5, 4),
        ("min_drdi_index", -2.41, 5),
        ("max_drdi_index", -1.99, 6),
        ("min_drdi_at_422pe", 1.39, 7),
        ("min_fraction_around_mode", 0.69, 8),
    ],
)
def test_relax_each_advanced_cut(sample_pd: dict, threshold: str, value: float, newly_accepted_run: int):
    data_filter = DataFilter(source_ra=10, source_dec=10, **{threshold: value})
    result = data_filter.apply_advanced_cuts(sample_pd["full_df"])
    assert result["run_number"].tolist() == [1, newly_accepted_run]


def test_plot_advanced_distributions_exposes_unfiltered_figures(sample_pd: dict):
    data_filter = DataFilter(source_ra=10, source_dec=10)
    statistics = sample_pd["full_df"].copy()
    statistics.loc[0, "n_subruns"] = 0  # Would be removed by the basic cuts.

    figures = plot_advanced_distributions(statistics, data_filter)

    assert set(figures) == set(DataFilter.ADVANCED_COLUMNS)
    assert all(isinstance(figure, Figure) for figure in figures.values())
    assert all(len(figure.axes) == 1 for figure in figures.values())
    assert all(
        sum(patch.get_height() for patch in figure.axes[0].patches) == len(statistics) for figure in figures.values()
    )

    for figure in figures.values():
        plt.close(figure)
