import pandas as pd
import pytest

from lst_ulities.datacheck import DataCheckStore, DataCheckTables, RunStatistics


def write_products(directory, run_number):
    tables = DataCheckTables(
        flatfield=pd.DataFrame({"runnumber": [run_number], "flatfield": [1.0]}),
        cosmics_intensity_spectrum=pd.DataFrame(
            {
                "runnumber": [run_number],
                "yyyymmdd": [20240101],
                "cosmics_intensity": [10.0],
            }
        ),
        runsummary=pd.DataFrame({"runnumber": [run_number], "n_events": [100]}),
    )
    tables.save_to_h5file(directory / "data_check_source.h5", overwrite=True)
    statistics = RunStatistics(pd.DataFrame({"date": [20240101]}, index=[run_number]))
    statistics.save_to_h5file(directory / "selected_runs_source.h5", overwrite=True)


def test_store_requires_explicit_initialization():
    store = DataCheckStore()

    assert not store.is_initialized
    with pytest.raises(RuntimeError, match="initialize_data_check"):
        _ = store.data_check_tables
    with pytest.raises(RuntimeError, match="initialize_data_check"):
        _ = store.run_statistics


def test_initialization_is_lazy_and_products_load_independently(tmp_path, monkeypatch):
    write_products(tmp_path, 1)
    table_loads = 0
    statistics_loads = 0
    original_tables_loader = DataCheckTables.from_files.__func__
    original_statistics_loader = RunStatistics.from_file.__func__

    def load_tables(cls, files):
        nonlocal table_loads
        table_loads += 1
        return original_tables_loader(cls, files)

    def load_statistics(cls, filename, key="run_statistics"):
        nonlocal statistics_loads
        statistics_loads += 1
        return original_statistics_loader(cls, filename, key)

    monkeypatch.setattr(DataCheckTables, "from_files", classmethod(load_tables))
    monkeypatch.setattr(RunStatistics, "from_file", classmethod(load_statistics))

    store = DataCheckStore().initialize(tmp_path)

    assert store.is_initialized
    assert store.directory == tmp_path
    assert table_loads == 0
    assert statistics_loads == 0

    assert list(store.data_check_tables.runsummary["runnumber"]) == [1]
    assert store.data_check_tables is store.data_check_tables
    assert table_loads == 1
    assert statistics_loads == 0

    assert list(store.run_statistics.df.index) == [1]
    assert store.run_statistics is store.run_statistics
    assert table_loads == 1
    assert statistics_loads == 1


def test_reinitialization_clears_both_lazy_caches(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_products(first, 1)
    write_products(second, 2)

    store = DataCheckStore().initialize(first)
    first_tables = store.data_check_tables
    first_statistics = store.run_statistics

    store.initialize(second)

    assert store.data_check_tables is not first_tables
    assert list(store.data_check_tables.runsummary["runnumber"]) == [2]
    assert store.run_statistics is not first_statistics
    assert list(store.run_statistics.df.index) == [2]


def test_store_supports_custom_product_names(tmp_path):
    write_products(tmp_path, 3)
    (tmp_path / "data_check_source.h5").rename(tmp_path / "tables.h5")
    (tmp_path / "selected_runs_source.h5").rename(tmp_path / "statistics.h5")

    store = DataCheckStore().initialize(
        tmp_path,
        data_check_pattern="tables.h5",
        run_statistics_pattern="statistics.h5",
    )

    assert list(store.data_check_tables.runsummary["runnumber"]) == [3]
    assert list(store.run_statistics.df.index) == [3]
