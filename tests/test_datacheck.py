import pandas as pd
import pytest

from lst_ulities.datacheck import DataCheckTables


@pytest.fixture
def sample_tables():
    """Build a minimal DataCheckTables instance for testing."""
    flatfield = pd.DataFrame(
        {
            "runnumber": [1, 1, 2, 2, 3, 3],
            "flatfield": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )
    cosmic_intensity_spectrum = pd.DataFrame(
        {
            "runnumber": [1, 2, 3],
            "yyyymmdd": [20240101, 20240102, 20240103],
            "cosmic_intensity": [10.0, 20.0, 30.0],
        }
    )
    runsummary = pd.DataFrame(
        {
            "runnumber": [1, 2, 3],
            "n_events": [100, 200, 300],
        }
    )
    return DataCheckTables(
        flatfield=flatfield,
        cosmic_intensity_spectrum=cosmic_intensity_spectrum,
        runsummary=runsummary,
    )


def test_select_runs_subset(sample_tables):
    """select_runs should keep only the requested runs and leave the original intact."""
    original_runs = set(sample_tables.cosmic_intensity_spectrum["runnumber"])

    filtered = sample_tables.select_runs([1, 3])

    assert set(filtered.flatfield["runnumber"]) == {1, 3}
    assert set(filtered.cosmic_intensity_spectrum["runnumber"]) == {1, 3}
    assert set(filtered.runsummary["runnumber"]) == {1, 3}
    assert len(filtered.flatfield) == 4
    assert len(filtered.cosmic_intensity_spectrum) == 2
    assert len(filtered.runsummary) == 2

    # Original object must be unchanged
    assert set(sample_tables.cosmic_intensity_spectrum["runnumber"]) == original_runs


def test_select_runs_with_set(sample_tables):
    """select_runs should accept a set of run numbers."""
    filtered = sample_tables.select_runs({2})

    assert set(filtered.flatfield["runnumber"]) == {2}
    assert list(filtered.cosmic_intensity_spectrum["runnumber"]) == [2]


def test_select_runs_empty_selection(sample_tables):
    """select_runs with no matching runs should return empty DataFrames."""
    filtered = sample_tables.select_runs([999])

    assert filtered.flatfield.empty
    assert filtered.cosmic_intensity_spectrum.empty
    assert filtered.runsummary.empty


def test_describe_output(sample_tables, capsys):
    """describe should print a summary containing run counts and date range."""
    sample_tables.describe()

    captured = capsys.readouterr().out
    assert "Number of runs" in captured
    assert "3" in captured
    assert "Run number range" in captured
    assert "1" in captured
    assert "3" in captured
    assert "Date range" in captured
    assert "2024-01-01" in captured
    assert "2024-01-03" in captured

    # Print the captured summary so it is visible when running with `pytest -s`.
    with capsys.disabled():
        print(captured)
