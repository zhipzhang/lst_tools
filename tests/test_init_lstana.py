import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lst_tools.pipeline import PipelineConfig, RunSelection
from lst_tools.scripts import init_lstana

MINIMAL_CONFIG = {
    "source": {"name": "Crab Nebula", "ra": 83.632, "dec": 22.0145},
    "zenith_binning": {"edges_deg": [0.0, 20.0, 30.0]},
    "data_filter": {"cuts": {"first_date": 20220101, "last_date": 20221231}},
}


class TestLoadDatacheckTables:
    def test_date_cuts_narrow_files_before_loading(self, monkeypatch):
        files = [
            "/data/DL1_datacheck_20211231.h5",
            "/data/DL1_datacheck_20220101.h5",
            "/data/DL1_datacheck_20220615.h5",
            "/data/DL1_datacheck_20221231.h5",
            "/data/DL1_datacheck_20230101.h5",
        ]
        monkeypatch.setattr(init_lstana, "glob_files", lambda dirs, pattern: files)
        captured = {}

        def fake_from_files(loaded_files):
            captured["files"] = list(loaded_files)
            return SimpleNamespace()

        monkeypatch.setattr(init_lstana.DataCheckTables, "from_files", fake_from_files)

        init_lstana.load_datacheck_tables(MINIMAL_CONFIG)

        assert captured["files"] == files[1:4]

    def test_no_files_in_range_raises(self, monkeypatch):
        monkeypatch.setattr(init_lstana, "glob_files", lambda dirs, pattern: ["/data/DL1_datacheck_20200101.h5"])

        with pytest.raises(FileNotFoundError, match="within the configured date range"):
            init_lstana.load_datacheck_tables(MINIMAL_CONFIG)


class FakeDataFilter:
    """Records its cuts and passes statistics through."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply_basic_cuts(self, statistics):
        return statistics

    def apply_advanced_cuts(self, statistics):
        return statistics


def make_stats_frame():
    return pd.DataFrame(
        {
            "run_number": [1, 2],
            "date": [20220426, 20220427],
            "mean_cos_zd": [np.cos(np.radians(15.0)), np.cos(np.radians(25.0))],
        }
    )


class TestSelectRuns:
    def test_returns_a_run_selection_with_bins_and_bookkeeping(self, monkeypatch):
        tables = SimpleNamespace(statistics=make_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)
        monkeypatch.setattr(init_lstana, "DataFilter", FakeDataFilter)

        selection = init_lstana.select_runs(MINIMAL_CONFIG)

        assert selection.edges == (0.0, 20.0, 30.0)
        assert selection.tables is tables
        assert selection.after_basic_cuts is tables.statistics
        assert [run.zenith_bin for run in selection] == ["zd_0_20", "zd_20_30"]
        # the zenith range of the configured binning becomes a data cut
        assert selection.data_filter.kwargs["min_zenith_angle"] == 0.0
        assert selection.data_filter.kwargs["max_zenith_angle"] == 30.0

    def test_advanced_cuts_apply_only_when_configured(self, monkeypatch):
        tables = SimpleNamespace(statistics=make_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)

        class CountingFilter(FakeDataFilter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.advanced_calls = 0

            def apply_advanced_cuts(self, statistics):
                self.advanced_calls += 1
                return statistics

        monkeypatch.setattr(init_lstana, "DataFilter", CountingFilter)
        selection = init_lstana.select_runs({**MINIMAL_CONFIG, "data_filter": {"with_advanced": True, "cuts": {}}})

        assert selection.data_filter.advanced_calls == 1


class TestBuildProducers:
    def test_producers_are_wired_to_each_others_output_dirs(self, tmp_path):
        pipeline_config = PipelineConfig.parse(
            {
                "dl1": {"link": {"output_directory": str(tmp_path / "linked_dl1")}},
                "dl2": {
                    "enabled": True,
                    "source": "build",
                    "build": {
                        "output_directory": str(tmp_path / "built_dl2"),
                        "rf_directory": "/models/nsb_tuning_$NSB_LEVEL/dec_347",
                    },
                },
                "dl3": {"enabled": True, "source": "link", "link": {"irf_type": "point"}},
            }
        )

        producers = init_lstana.make_producers(pipeline_config, MINIMAL_CONFIG, tmp_path / "out")

        _, dl2, dl3 = producers
        assert dl2.dl1_directory == tmp_path / "linked_dl1"
        assert dl2.output_dir == tmp_path / "built_dl2"
        assert dl3.dl2_directory == tmp_path / "built_dl2"
        assert dl3.source.name == "Crab Nebula"

    def test_producers_default_to_the_output_root_tree(self, tmp_path):
        pipeline_config = PipelineConfig.parse({"dl1": {}, "dl2": {"enabled": True, "source": "link"}})

        producers = init_lstana.make_producers(pipeline_config, MINIMAL_CONFIG, tmp_path / "out")

        dl1, dl2, dl3 = producers
        assert dl1.output_dir == tmp_path / "out" / "dl1"
        assert dl2.dl1_directory == tmp_path / "out" / "dl1"
        assert dl3.dl2_directory == tmp_path / "out" / "dl2"


class TestMain:
    def test_disabled_producers_are_skipped(self, tmp_path, monkeypatch, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent(
                """
                [source]
                ra = 83.632
                dec = 22.0145
                name = "Crab Nebula"

                [zenith_binning]
                edges_deg = [0.0, 20.0]

                [data_filter.cuts]
                first_date = 20220101
                last_date = 20221231

                [dl1]
                enabled = false
                """
            )
        )
        empty_selection = RunSelection(
            edges=(0.0, 20.0),
            selected=pd.DataFrame({"run_number": [], "date": [], "zenith_bin": []}),
        )
        monkeypatch.setattr(init_lstana, "select_runs", lambda raw_config: empty_selection)
        monkeypatch.setattr(init_lstana, "save_datacheck_outputs", lambda *args: None)
        produced = []
        disabled_producer = SimpleNamespace(
            config=SimpleNamespace(enabled=False),
            produce=lambda selection: produced.append(1),
        )
        monkeypatch.setattr(
            init_lstana, "make_producers", lambda pipeline_config, raw_config, output_root: [disabled_producer]
        )
        monkeypatch.setattr("sys.argv", ["init_lstana", str(config_path), "-o", str(tmp_path / "out")])

        init_lstana.main()

        assert produced == []
        assert "Selected 0 runs" in capsys.readouterr().out
