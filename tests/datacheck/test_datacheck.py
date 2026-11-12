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


def test_from_files_warns_and_skips_incomplete_files(sample_tables: DataCheckTables, tmp_path, caplog):
    valid_file = tmp_path / "valid.h5"
    incomplete_file = tmp_path / "incomplete.h5"
    sample_tables.save_to_h5file(valid_file, overwrite=True)
    sample_tables.save_to_h5file(incomplete_file, overwrite=True)
    with pd.HDFStore(incomplete_file, mode="a") as store:
        store.remove("flatfield")

    with caplog.at_level("WARNING", logger="lst_tools.datacheck.datacheck"):
        loaded = DataCheckTables.from_files([str(incomplete_file), str(valid_file)])

    assert loaded.runsummary["runnumber"].tolist() == [1, 2]
    assert str(incomplete_file) in caplog.text
    assert "/flatfield" in caplog.text


def test_from_files_concatenates_complete_files(sample_tables: DataCheckTables, tmp_path):
    first_file = tmp_path / "first.h5"
    second_file = tmp_path / "second.h5"
    sample_tables.select_runs([1]).save_to_h5file(first_file, overwrite=True)
    sample_tables.select_runs([2]).save_to_h5file(second_file, overwrite=True)

    loaded = DataCheckTables.from_files([str(second_file), str(first_file)])

    assert loaded.flatfield["runnumber"].tolist() == [2, 2, 1, 1]
    assert loaded.cosmics_intensity_spectrum["runnumber"].tolist() == [2, 2, 1, 1]
    assert loaded.runsummary["runnumber"].tolist() == [2, 1]


def test_from_files_rejects_empty_input():
    with pytest.raises(ValueError, match="No valid DataCheck files"):
        DataCheckTables.from_files([])


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
    assert statistics["runnumber"].to_list() == [1, 2]
    assert statistics["n_subruns"].to_list() == [2, 2]
    assert statistics["n_flatfield"].to_list() == [30, 40]


def test_get_statistics_rejects_unknown_table(sample_tables: DataCheckTables):
    with pytest.raises(ValueError, match="Unknown DataCheck table.*missing"):
        sample_tables.get_statistics({"missing": {"n_subruns": ("runnumber", "size")}})


def test_get_statistics_reports_missing_source_columns(sample_tables: DataCheckTables):
    spec = {"runsummary": {"n_pedestal": ("num_pedestals", "first")}}

    with pytest.raises(ValueError, match="runsummary.*num_pedestals"):
        sample_tables.get_statistics(spec)


def test_get_statistics_preserves_runs_missing_from_one_table(sample_tables: DataCheckTables):
    sample_tables.runsummary = sample_tables.runsummary.loc[lambda table: table["runnumber"] == 1]
    spec = {
        "cosmics_intensity_spectrum": {"n_subruns": ("runnumber", "size")},
        "runsummary": {"n_flatfield": ("num_flatfield", "first")},
    }

    statistics = sample_tables.get_statistics(spec)

    expected = pd.DataFrame(
        {
            "runnumber": [1, 2],
            "n_subruns": [2, 2],
            "n_flatfield": [30.0, float("nan")],
        }
    )
    pd.testing.assert_frame_equal(statistics, expected)


def test_select_runs(sample_tables: DataCheckTables):
    selected = sample_tables.select_runs([2])
    assert selected is not None
    assert selected.cosmics_intensity_spectrum["runnumber"].to_list() == [2, 2]
    assert selected.runsummary["runnumber"].to_list() == [2]
    assert selected.runsummary["n_events"].to_list() == [200]


def test_empty_selection_round_trips_through_hdf5(sample_tables: DataCheckTables, tmp_path):
    output_file = tmp_path / "empty_datacheck.h5"
    sample_tables.select_runs([]).save_to_h5file(output_file, overwrite=True)

    loaded = DataCheckTables.from_files([str(output_file)])

    assert loaded.flatfield.empty
    assert loaded.cosmics_intensity_spectrum.empty
    assert loaded.runsummary.empty

    sample_tables.save_to_h5file(output_file)
    reloaded = DataCheckTables.from_files([str(output_file)])
    assert reloaded.flatfield["runnumber"].tolist() == [1, 1, 2, 2]
    assert reloaded.cosmics_intensity_spectrum["runnumber"].tolist() == [1, 1, 2, 2]
    assert reloaded.runsummary["runnumber"].tolist() == [1, 2]
