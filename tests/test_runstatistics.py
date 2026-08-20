import numpy as np
import pandas as pd
import pytest

from lst_ulities.datacheck import DataCheckTables, RunStatistics


@pytest.fixture
def sample_tables():
    """Build a minimal DataCheckTables instance for testing."""
    flatfield = pd.DataFrame(
        {
            "runnumber": [1, 1, 2, 2, 3, 3],
            "flatfield": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )
    cosmics_intensity_spectrum = pd.DataFrame(
        {
            "runnumber": [1, 1, 2, 2, 3, 3],
            "yyyymmdd": [20240101, 20240101, 20240102, 20240102, 20240103, 20240103],
            "cosmics_intensity": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
        }
    )
    runsummary = pd.DataFrame(
        {
            "runnumber": [1, 2, 3],
            "n_events": [100, 200, 300],
            "source_name": ["Crab", "Crab", "Mrk421"],
        }
    )
    return DataCheckTables(
        flatfield=flatfield,
        cosmics_intensity_spectrum=cosmics_intensity_spectrum,
        runsummary=runsummary,
    )


def test_from_tables_mean_and_std(sample_tables):
    """from_tables should compute mean and std per runnumber."""
    spec = {
        "cosmics_intensity_spectrum": {
            "mean_intensity": ("cosmics_intensity", "mean"),
            "std_intensity": ("cosmics_intensity", "std"),
        },
    }

    stats = RunStatistics.from_tables(sample_tables, spec=spec)

    assert list(stats.run_numbers) == [1, 2, 3]
    assert stats.df.loc[1, "mean_intensity"] == pytest.approx(11.0)
    assert stats.df.loc[2, "mean_intensity"] == pytest.approx(21.0)
    assert stats.df.loc[3, "mean_intensity"] == pytest.approx(31.0)
    assert stats.df.loc[1, "std_intensity"] == pytest.approx(np.std([10.0, 12.0], ddof=1))


def test_from_tables_first_and_lambda(sample_tables):
    """from_tables should support 'first' and custom lambda aggregations."""
    spec = {
        "runsummary": {
            "first_source": ("source_name", "first"),
            "total_events": ("n_events", "sum"),
        },
        "flatfield": {
            "max_flatfield": ("flatfield", lambda x: x.max()),
        },
    }

    stats = RunStatistics.from_tables(sample_tables, spec=spec)

    assert list(stats.run_numbers) == [1, 2, 3]
    assert list(stats.df["first_source"]) == ["Crab", "Crab", "Mrk421"]
    assert stats.df.loc[1, "total_events"] == 100
    assert stats.df.loc[1, "max_flatfield"] == pytest.approx(1.1)
    assert stats.df.loc[2, "max_flatfield"] == pytest.approx(2.1)


def test_from_tables_outer_join(sample_tables):
    """from_tables should outer-join statistics from different tables."""
    # Remove run 3 from runsummary so we can test outer join behavior.
    tables = DataCheckTables(
        flatfield=sample_tables.flatfield,
        cosmics_intensity_spectrum=sample_tables.cosmics_intensity_spectrum,
        runsummary=sample_tables.runsummary[sample_tables.runsummary["runnumber"] != 3].reset_index(drop=True),
    )
    spec = {
        "cosmics_intensity_spectrum": {
            "mean_intensity": ("cosmics_intensity", "mean"),
        },
        "runsummary": {
            "first_source": ("source_name", "first"),
        },
    }

    stats = RunStatistics.from_tables(tables, spec=spec)

    assert list(stats.run_numbers) == [1, 2, 3]
    assert pd.notna(stats.df.loc[3, "mean_intensity"])
    assert pd.isna(stats.df.loc[3, "first_source"])


def test_select_subset(sample_tables):
    """select should return a new RunStatistics with only the masked runs."""
    spec = {
        "cosmics_intensity_spectrum": {
            "mean_intensity": ("cosmics_intensity", "mean"),
        },
    }
    stats = RunStatistics.from_tables(sample_tables, spec=spec)

    mask = stats.df["mean_intensity"] > 15.0
    selected = stats.select(mask)

    assert list(selected.run_numbers) == [2, 3]
    assert list(stats.run_numbers) == [1, 2, 3]


def test_run_numbers_property(sample_tables):
    """run_numbers should expose the DataFrame index."""
    spec = {
        "runsummary": {
            "total_events": ("n_events", "sum"),
        },
    }
    stats = RunStatistics.from_tables(sample_tables, spec=spec)

    assert isinstance(stats.run_numbers, pd.Index)
    assert list(stats.run_numbers) == [1, 2, 3]


def test_from_tables_custom_named_aggregation(sample_tables):
    """A user-defined callable should produce the expected named column."""
    spec = {
        "flatfield": {
            "flatfield_range": ("flatfield", lambda x: x.max() - x.min()),
        },
    }

    stats = RunStatistics.from_tables(sample_tables, spec=spec)

    assert stats.df.loc[1, "flatfield_range"] == pytest.approx(0.1)
    assert stats.df.loc[2, "flatfield_range"] == pytest.approx(0.1)
    assert stats.df.loc[3, "flatfield_range"] == pytest.approx(0.1)


def test_assign_zenith_bins_has_unique_boundaries():
    angles = np.array([0.0, 19.999, 20.0, 39.999, 40.0, 70.0])
    stats = RunStatistics(pd.DataFrame({"mean_cos_zd": np.cos(np.radians(angles))}))

    result = stats.assign_zenith_bins([0, 20, 40, 70])

    assert result.df["zenith_bin"].tolist() == [
        "zd_0_20",
        "zd_0_20",
        "zd_20_40",
        "zd_20_40",
        "zd_40_70",
        "zd_40_70",
    ]
    assert result.df["mean_zenith_angle"].to_numpy() == pytest.approx(angles)


@pytest.mark.parametrize(
    "edges",
    [[], [20], [0, 20, 20], [0, 30, 20], [-1, 20], [0, 91], [0, np.nan, 20]],
)
def test_assign_zenith_bins_rejects_invalid_edges(edges):
    stats = RunStatistics(pd.DataFrame({"mean_cos_zd": [1.0]}))

    with pytest.raises(ValueError):
        stats.assign_zenith_bins(edges)
