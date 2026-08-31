import pandas as pd
import pytest

from lst_tools.datacheck import DataCheckTables


@pytest.fixture
def sample_tables():
    """Build a minimal DataCheckTables instance for testing."""
    flatfield = pd.DataFrame(
        {
            "runnumber": [1, 1, 2, 2],
        }
    )
    cosmics_intensity_spectrum = pd.DataFrame(
        {
            "runnumber": [1, 1, 2, 2],
            "yyyymmdd": [20240101, 20240101, 20240102, 20240102],
            "cosmics_intensity": [10.0, 20.0, 30.0, 40],
        }
    )
    runsummary = pd.DataFrame(
        {
            "runnumber": [1, 2],
            "n_events": [100, 200],
            "num_flatfield": [30, 40],
        }
    )
    return DataCheckTables(
        flatfield=flatfield,
        cosmics_intensity_spectrum=cosmics_intensity_spectrum,
        runsummary=runsummary,
    )


def test_get_statistics(sample_tables: DataCheckTables):
    """
    Test the way to compute statistics from the sample_tables
    """
    simple_spec = {
        "cosmics_intensity_spectrum": {
            "n_subruns": ("runnumber", "size"),
        },
        "runsummary": {"n_flatfield": ("num_flatfield", "first")},
    }

    statistics = sample_tables.get_statistics(simple_spec)
    assert statistics["run_number"].to_list() == [1, 2]
    assert statistics["n_subruns"].to_list() == [2, 2]
    assert statistics["n_flatfield"].to_list() == [30, 40]


def test_get_statistics_preserves_runs_missing_from_one_table(sample_tables: DataCheckTables):
    sample_tables.runsummary = sample_tables.runsummary.loc[lambda table: table["runnumber"] == 1]
    spec = {
        "cosmics_intensity_spectrum": {"n_subruns": ("runnumber", "size")},
        "runsummary": {"n_flatfield": ("num_flatfield", "first")},
    }

    statistics = sample_tables.get_statistics(spec)

    assert statistics["run_number"].to_list() == [1, 2]
    assert pd.isna(statistics.loc[1, "n_flatfield"])


def test_statistics_uses_default_spec(sample_tables: DataCheckTables, monkeypatch):
    from lst_tools.datacheck import datacheck

    default_spec = {
        "cosmics_intensity_spectrum": {
            "n_subruns": ("runnumber", "size"),
        },
        "runsummary": {
            "n_flatfield": ("num_flatfield", "first"),
        },
    }
    monkeypatch.setattr(datacheck, "DEFAULT_SPEC", default_spec)

    pd.testing.assert_frame_equal(
        sample_tables.statistics,
        sample_tables.get_statistics(default_spec),
    )


def test_select_runs(sample_tables: DataCheckTables):
    selected = sample_tables.select_runs([2])
    assert selected is not None
    assert selected.cosmics_intensity_spectrum["runnumber"].to_list() == [2, 2]
    assert selected.runsummary["runnumber"].to_list() == [2]
    assert selected.runsummary["n_events"].to_list() == [200]
