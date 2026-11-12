import textwrap
from collections import Counter
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

import lst_tools.catalog
from lst_tools.catalog import CatalogSource
from lst_tools.datacheck import DataCheckTables
from lst_tools.pipeline import PipelineConfig, RunSelection
from lst_tools.scripts import init_lstana

MINIMAL_CONFIG = {
    "source": {"name": "Crab Nebula", "ra": 83.632, "dec": 22.0145},
    "data_filter": {
        "cuts": {
            "first_date": 20220101,
            "last_date": 20221231,
            "min_zenith_angle": 0.0,
            "max_zenith_angle": 30.0,
        }
    },
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
    def test_returns_all_selected_runs_and_bookkeeping(self, monkeypatch):
        tables = SimpleNamespace(statistics=make_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)
        monkeypatch.setattr(init_lstana, "DataFilter", FakeDataFilter)

        selection = init_lstana.select_runs(MINIMAL_CONFIG)

        assert selection.tables is tables
        assert selection.after_basic_cuts is tables.statistics
        assert [run.run_number for run in selection] == [1, 2]
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


def make_basic_stats_frame():
    """Two runs pointing (RA 10, dec 0) and (RA 30, dec 0), both in the bins."""
    return pd.DataFrame(
        {
            "run_number": [1, 2],
            "n_subruns": [2, 2],
            "date": [20240101, 20240101],
            "n_flatfield": [1, 1],
            "n_pedestal": [1, 1],
            "mean_ra": [10.0, 30.0],
            "mean_dec": [0.0, 0.0],
            "mean_cos_zd": [np.cos(np.radians(15.0)), np.cos(np.radians(25.0))],
            "observation_time": [1800.0, 1800.0],
            "pointing_dec_std": [0.001, 0.001],
        }
    )


class TestAwaySourceSelection:
    def test_rejects_pointings_near_sources_and_disables_target_angle_cut(self, monkeypatch):
        tables = SimpleNamespace(statistics=make_basic_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)
        near_run_2 = (CatalogSource(name="near-run-2", coord=SkyCoord(ra=30, dec=0, unit="deg"), catalog="test"),)
        monkeypatch.setattr(lst_tools.catalog, "load_tev_sources", lambda: near_run_2)

        selection = init_lstana.select_runs({**MINIMAL_CONFIG, "data_filter": {"away_source": True}})

        # Run 1 is far from the target, so it only survives because the
        # 0.3–0.5 degree target-angle cut was disabled; run 2 points at the
        # catalog source and is rejected.
        assert selection.selected["run_number"].tolist() == [1]
        assert selection.data_filter.min_angle_to_source == 0.0
        assert selection.data_filter.max_angle_to_source == 180.0

    def test_without_away_source_the_target_angle_cut_still_applies(self, monkeypatch):
        tables = SimpleNamespace(statistics=make_basic_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)

        selection = init_lstana.select_runs(MINIMAL_CONFIG)

        assert selection.selected["run_number"].tolist() == []
        assert selection.data_filter.max_angle_to_source == 0.5

    @pytest.mark.parametrize(
        ("data_filter", "min_distance", "extension_factor"),
        [
            ({"away_source": True}, 3.0, 2.5),
            ({"away_source": True, "min_distance": 0.7, "extension_factor": 4.0}, 0.7, 4.0),
        ],
    )
    def test_distance_options_reach_the_catalog_separation(
        self, monkeypatch, data_filter, min_distance, extension_factor
    ):
        tables = SimpleNamespace(statistics=make_stats_frame())
        monkeypatch.setattr(init_lstana, "load_datacheck_tables", lambda config: tables)
        monkeypatch.setattr(init_lstana, "DataFilter", FakeDataFilter)
        monkeypatch.setattr(lst_tools.catalog, "load_tev_sources", lambda: ())
        captured = {}

        def fake_away(statistics, sources, *, min_separation, extension_factor):
            captured["min_separation"] = min_separation
            captured["extension_factor"] = extension_factor
            return statistics

        monkeypatch.setattr(init_lstana, "select_runs_away_from_sources", fake_away)

        init_lstana.select_runs({**MINIMAL_CONFIG, "data_filter": data_filter})

        assert captured["min_separation"].to_value(u.deg) == min_distance
        assert captured["extension_factor"] == extension_factor


def make_datacheck_tables():
    """Datacheck tables carrying every column the default statistics need."""
    runs = [1, 2, 3]
    return DataCheckTables(
        flatfield=pd.DataFrame({"runnumber": runs, "num_flatfield": [1, 1, 1]}),
        cosmics_intensity_spectrum=pd.DataFrame(
            {
                "runnumber": runs,
                "yyyymmdd": [20240101, 20240101, 20240102],
                "ZD_corrected_cosmics_rate_at_422_pe": [10.0, 11.0, 12.0],
                "intensity_at_reference_rate": [100.0, 110.0, 120.0],
                "light_yield": [1.0, 1.1, 1.2],
                "ZD_corrected_cosmics_spectral_index": [-2.6, -2.5, -2.4],
                "intensity_spectrum_fit_p_value": [0.5, 0.6, 0.7],
                "ZD_corrected_intensity_at_half_peak_rate": [50.0, 51.0, 52.0],
                "ra_tel": [83.6, 83.7, 83.8],
                "dec_tel": [22.0, 22.0, 22.0],
                "cos_zenith": [np.cos(np.radians(z)) for z in (15.0, 25.0, 35.0)],
                "diffuse_nsb_std": [0.1, 0.1, 0.1],
            }
        ),
        runsummary=pd.DataFrame(
            {
                "runnumber": runs,
                "num_flatfield": [1, 1, 1],
                "num_pedestals": [1, 1, 1],
                "elapsed_time": [1800.0, 1800.0, 1800.0],
            }
        ),
    )


class TestSavedSelection:
    def test_saved_outputs_reload_as_the_same_selection(self, tmp_path, monkeypatch):
        selection = RunSelection(
            selected=pd.DataFrame({"run_number": [1, 2], "date": [20240101, 20240101]}),
            tables=make_datacheck_tables(),
            after_basic_cuts=pd.DataFrame(),
        )
        monkeypatch.setattr("lst_tools.datacheck.plot_advanced_distributions", lambda *args: {})

        init_lstana.save_datacheck_outputs(selection, MINIMAL_CONFIG, tmp_path / "out" / "data_check")

        loaded = init_lstana.load_saved_selection(MINIMAL_CONFIG, tmp_path / "out")

        assert [run.run_number for run in loaded] == [1, 2]
        assert loaded.after_basic_cuts is None

    def test_absent_file_loads_nothing(self, tmp_path):
        assert init_lstana.load_saved_selection(MINIMAL_CONFIG, tmp_path / "out") is None

    def test_refresh_flag_skips_the_saved_selection(self, tmp_path, monkeypatch):
        path = init_lstana.selection_file(MINIMAL_CONFIG, tmp_path / "out")
        path.parent.mkdir(parents=True)
        path.touch()
        fresh = RunSelection(selected=pd.DataFrame({"run_number": [], "date": []}))
        recomputed = []
        monkeypatch.setattr(init_lstana, "select_runs", lambda config: recomputed.append(config) or fresh)
        monkeypatch.setattr(init_lstana, "save_datacheck_outputs", lambda *args: None)
        monkeypatch.setattr(init_lstana, "make_producers", lambda *args, stage="all": [])
        monkeypatch.setattr(
            "sys.argv",
            ["init_lstana", str(self.write_config(tmp_path)), "-o", str(tmp_path / "out"), "--refresh-selection"],
        )

        init_lstana.main()

        assert len(recomputed) == 1

    @staticmethod
    def write_config(tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent(
                """
                [source]
                ra = 83.632
                dec = 22.0145
                name = "Crab Nebula"

                [data_filter.cuts]
                first_date = 20220101
                last_date = 20221231
                """
            )
        )
        return config_path

    def test_saved_selection_is_used_by_main(self, tmp_path, monkeypatch, capsys):
        path = init_lstana.selection_file(MINIMAL_CONFIG, tmp_path / "out")
        path.parent.mkdir(parents=True)
        path.touch()
        monkeypatch.setattr(
            init_lstana.DataCheckTables,
            "from_files",
            lambda files: SimpleNamespace(statistics=make_stats_frame()),
        )
        monkeypatch.setattr(
            init_lstana,
            "select_runs",
            lambda config: pytest.fail("an existing selection must not be recomputed"),
        )
        monkeypatch.setattr(init_lstana, "make_producers", lambda *args, stage="all": [])
        monkeypatch.setattr("sys.argv", ["init_lstana", str(self.write_config(tmp_path)), "-o", str(tmp_path / "out")])

        init_lstana.main()

        assert "Loaded the saved run selection" in capsys.readouterr().out


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

    @pytest.mark.parametrize("stage,expected_level", [("dl1", "dl1"), ("dl2", "dl2"), ("dl3", "dl3")])
    def test_stage_filters_producers(self, tmp_path, stage, expected_level):
        pipeline_config = PipelineConfig.parse({"dl1": {}, "dl2": {"enabled": True, "source": "link"}})

        producers = init_lstana.make_producers(pipeline_config, MINIMAL_CONFIG, tmp_path / "out", stage=stage)

        assert len(producers) == 1
        assert producers[0].level == expected_level

    def test_stage_all_returns_all_producers(self, tmp_path):
        pipeline_config = PipelineConfig.parse({"dl1": {}, "dl2": {"enabled": True, "source": "link"}})

        producers = init_lstana.make_producers(pipeline_config, MINIMAL_CONFIG, tmp_path / "out", stage="all")

        assert [p.level for p in producers] == ["dl1", "dl2", "dl3"]


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

                [data_filter.cuts]
                first_date = 20220101
                last_date = 20221231

                [dl1]
                enabled = false
                """
            )
        )
        empty_selection = RunSelection(selected=pd.DataFrame({"run_number": [], "date": []}))
        monkeypatch.setattr(init_lstana, "select_runs", lambda raw_config: empty_selection)
        monkeypatch.setattr(init_lstana, "save_datacheck_outputs", lambda *args: None)
        produced = []
        disabled_producer = SimpleNamespace(
            config=SimpleNamespace(enabled=False),
            produce=lambda selection: produced.append(1),
        )
        monkeypatch.setattr(
            init_lstana,
            "make_producers",
            lambda pipeline_config, raw_config, output_root, stage="all": [disabled_producer],
        )
        monkeypatch.setattr("sys.argv", ["init_lstana", str(config_path), "-o", str(tmp_path / "out")])

        init_lstana.main()

        assert produced == []
        assert "Selected 0 runs" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "stage,should_save_datacheck", [("all", True), ("dl1", True), ("dl2", False), ("dl3", False)]
    )
    def test_stage_flag_controls_datacheck_output(self, tmp_path, monkeypatch, stage, should_save_datacheck):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent(
                """
                [source]
                ra = 83.632
                dec = 22.0145
                name = "Crab Nebula"

                [data_filter.cuts]
                first_date = 20220101
                last_date = 20221231
                """
            )
        )
        empty_selection = RunSelection(selected=pd.DataFrame({"run_number": [], "date": []}))
        monkeypatch.setattr(init_lstana, "select_runs", lambda raw_config: empty_selection)

        datacheck_calls = []
        monkeypatch.setattr(init_lstana, "save_datacheck_outputs", lambda *args: datacheck_calls.append(args))

        requested_stages = []
        monkeypatch.setattr(
            init_lstana,
            "make_producers",
            lambda pipeline_config, raw_config, output_root, stage="all": requested_stages.append(stage) or [],
        )

        argv = ["init_lstana", str(config_path), "-o", str(tmp_path / "out")]
        if stage != "all":
            argv.extend(["--stage", stage])
        monkeypatch.setattr("sys.argv", argv)

        init_lstana.main()

        assert len(datacheck_calls) == (1 if should_save_datacheck else 0)
        assert requested_stages == [stage]

    def test_main_stops_at_slurm_jobs(self, tmp_path, monkeypatch, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent(
                """
                [source]
                ra = 83.632
                dec = 22.0145
                name = "Crab Nebula"

                [data_filter.cuts]
                first_date = 20220101
                last_date = 20221231
                """
            )
        )
        empty_selection = RunSelection(selected=pd.DataFrame({"run_number": [], "date": []}))
        monkeypatch.setattr(init_lstana, "select_runs", lambda raw_config: empty_selection)
        monkeypatch.setattr(init_lstana, "save_datacheck_outputs", lambda *args: None)

        dl2_called = []
        dl3_called = []
        p_dl2 = SimpleNamespace(
            level="dl2",
            config=SimpleNamespace(enabled=True),
            produce=lambda selection: dl2_called.append(1) or Counter({"dl2:slurm_jobs": 2}),
        )
        p_dl3 = SimpleNamespace(
            level="dl3",
            config=SimpleNamespace(enabled=True),
            produce=lambda selection: dl3_called.append(1) or Counter(),
        )

        monkeypatch.setattr(
            init_lstana,
            "make_producers",
            lambda pipeline_config, raw_config, output_root, stage="all": [p_dl2, p_dl3],
        )

        monkeypatch.setattr("sys.argv", ["init_lstana", str(config_path), "-o", str(tmp_path / "out")])

        init_lstana.main()

        assert len(dl2_called) == 1
        assert len(dl3_called) == 0
        stdout = capsys.readouterr().out
        assert "Stopped pipeline after dl2" in stdout
