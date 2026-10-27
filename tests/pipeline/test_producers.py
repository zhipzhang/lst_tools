from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import lst_tools.pipeline.producers as producers_module
from lst_tools.event_filter import EventFilter
from lst_tools.finder.datalevel_attributes import DL3Attributes, DLAttributes
from lst_tools.pipeline import (
    DL1LinkOptions,
    DL1Producer,
    DL1ProducerConfig,
    DL2BuildOptions,
    DL2LinkOptions,
    DL2Producer,
    DL2ProducerConfig,
    DL3BuildOptions,
    DL3LinkOptions,
    DL3Producer,
    DL3ProducerConfig,
    RunSelection,
    SourceConfig,
)


def make_selection(rows, edges=(0.0, 20.0, 30.0)) -> RunSelection:
    return RunSelection(edges=edges, selected=pd.DataFrame(rows))


def make_dl_attrs(path: str, run_number: int = 10623) -> DLAttributes:
    return DLAttributes(
        path=Path(path),
        run_number=run_number,
        tailcuts_level=(10, 5),
        lstchain_version="0.11",
        nsb_level=None,
    )


def make_dl3_attrs(
    path: str,
    irf_type: str = "full-enclosure",
    mc_data_type: str = "ring-wobble",
    gh_eff: float = 0.7,
    theta_containment: float = 0.7,
    lstchain_version: str = "0.11",
) -> DL3Attributes:
    return DL3Attributes(
        path=Path(path),
        run_number=14083,
        tailcuts_level=(10, 5),
        lstchain_version=lstchain_version,
        nsb_level=None,
        irf_type=irf_type,
        mc_data_type=mc_data_type,
        gh_eff=gh_eff,
        theta_containment=theta_containment,
    )


def patch_finder(monkeypatch, results_by_key: dict):
    """Fake find_la_palma_files returning a per-(date, run, level) result.

    Each result needs only the attributes the producer under test reads:
    `newest_attributes` (single attrs or list, per data level).
    """

    def fake_find(date, run_number, level):
        return results_by_key[(date, run_number, level)]

    monkeypatch.setattr(producers_module, "find_la_palma_files", fake_find)


RUN_10623 = {"run_number": 10623, "date": 20220426, "zenith_bin": "zd_20_30"}
RUN_10624 = {"run_number": 10624, "date": 20220426, "zenith_bin": "zd_0_20"}
RUN_UNBINNED = {"run_number": 10625, "date": 20220427, "zenith_bin": None}


class TestDL1Producer:
    def test_link_places_newest_files_in_zenith_bin_tree(self, tmp_path, monkeypatch):
        source_1 = tmp_path / "server" / "dl1_LST-1.Run10623.h5"
        source_2 = tmp_path / "server" / "dl1_LST-1.Run10624.h5"
        source_1.parent.mkdir(parents=True)
        source_1.touch()
        source_2.touch()
        patch_finder(
            monkeypatch,
            {
                (20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source_1)),
                (20220426, 10624, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source_2)),
            },
        )
        producer = DL1Producer(DL1ProducerConfig(), tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert (tmp_path / "out" / "dl1" / "zd_20_30" / source_1.name).is_symlink()
        assert (tmp_path / "out" / "dl1" / "zd_0_20" / source_2.name).is_symlink()
        assert counts["dl1:created"] == 2

    def test_link_is_idempotent(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl1_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        producer = DL1Producer(DL1ProducerConfig(), tmp_path / "out")
        selection = make_selection([RUN_10623])

        first = producer.produce(selection)
        second = producer.produce(selection)

        assert first["dl1:created"] == 1
        assert second["dl1:existing"] == 1
        assert second["dl1:created"] == 0

    def test_link_conflict_warns_and_keeps_existing(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl1_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        output_root = tmp_path / "out"
        taken = output_root / "dl1" / "zd_20_30" / source.name
        taken.parent.mkdir(parents=True)
        taken.symlink_to(tmp_path / "somewhere_else.h5")
        producer = DL1Producer(DL1ProducerConfig(), output_root)

        with pytest.warns(UserWarning, match="points elsewhere"):
            counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl1:conflict"] == 1
        assert counts["dl1:created"] == 0

    def test_link_missing_server_file_skips_run(self, tmp_path, monkeypatch):
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=None)})
        producer = DL1Producer(DL1ProducerConfig(), tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl1:created"] == 0
        assert not (tmp_path / "out" / "dl1").exists()

    def test_unbinned_runs_warn_and_are_counted(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl1_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        producer = DL1Producer(DL1ProducerConfig(), tmp_path / "out")

        with pytest.warns(UserWarning, match="no zenith bin"):
            counts = producer.produce(make_selection([RUN_10623, RUN_UNBINNED]))

        assert counts["dl1:unbinned"] == 1
        assert counts["dl1:created"] == 1

    def test_build_raises_not_implemented_yet(self, tmp_path):
        config = DL1ProducerConfig(enabled=True, source="build")  # type: ignore[arg-type]

        with pytest.raises(NotImplementedError, match="dl1: building is not supported yet"):
            DL1Producer(config, tmp_path).produce(make_selection([RUN_10623]))

    def test_link_output_directory_override(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl1_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        config = DL1ProducerConfig(link=DL1LinkOptions(output_directory=tmp_path / "custom_dl1"))
        producer = DL1Producer(config, tmp_path / "out")

        producer.produce(make_selection([RUN_10623]))

        assert (tmp_path / "custom_dl1" / "zd_20_30" / source.name).is_symlink()
        assert not (tmp_path / "out" / "dl1").exists()

    def test_disabled_producer_does_nothing(self, tmp_path, monkeypatch):
        def forbidden_find(date, run_number, level):  # pragma: no cover - must not be called
            raise AssertionError("a disabled producer must not search the server")

        monkeypatch.setattr(producers_module, "find_la_palma_files", forbidden_find)
        producer = DL1Producer(DL1ProducerConfig(enabled=False), tmp_path / "out")

        assert producer.produce(make_selection([RUN_10623])) == {}


class TestDL2Producer:
    def test_link_uses_same_tree_as_dl1(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl2_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl2"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        producer = DL2Producer(DL2ProducerConfig(enabled=True, source="link"), tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert (tmp_path / "out" / "dl2" / "zd_20_30" / source.name).is_symlink()
        assert counts["dl2:created"] == 1

    @pytest.mark.parametrize(
        "source, expected_dir",
        [
            ("link", "linked_dl2"),
            ("build", "built_dl2"),
        ],
    )
    def test_output_dir_follows_source_mode(self, tmp_path, source, expected_dir):
        config = DL2ProducerConfig(
            enabled=True,
            source=source,
            link=DL2LinkOptions(output_directory=tmp_path / "linked_dl2"),
            build=DL2BuildOptions(
                output_directory=tmp_path / "built_dl2",
                rf_directory="/models/nsb_tuning_$NSB_LEVEL/dec_347",
            ),
        )
        producer = DL2Producer(config, tmp_path / "out")

        assert producer.output_dir == tmp_path / expected_dir

    def test_output_dir_defaults_below_output_root(self, tmp_path):
        producer = DL2Producer(DL2ProducerConfig(enabled=True, source="link"), tmp_path / "out")

        assert producer.output_dir == tmp_path / "out" / "dl2"

    def _make_dl1_tree(self, tmp_path, *run_numbers: int) -> list[Path]:
        files = []
        for run_number in run_numbers:
            dl1_file = tmp_path / "out" / "dl1" / "zd_20_30" / f"dl1_LST-1.Run{run_number:05d}.h5"
            dl1_file.parent.mkdir(parents=True, exist_ok=True)
            dl1_file.touch()
            files.append(dl1_file)
        return files

    def _fake_batch(self, monkeypatch):
        """dl1_to_dl2_batch stand-in: records calls, creates the DL2 files."""
        calls = []

        def fake_batch(files, out_root, model_directory, overwrite=False):
            out_root = Path(out_root)
            out_root.mkdir(parents=True, exist_ok=True)
            calls.append((list(files), out_root, Path(model_directory), overwrite))
            for dl1_file in files:
                (out_root / Path(dl1_file).name.replace("dl1", "dl2", 1)).touch()

        monkeypatch.setattr(producers_module, "dl1_to_dl2_batch", fake_batch)
        return calls

    def test_build_reconstructs_pending_dl1(self, tmp_path, monkeypatch):
        (dl1_file,) = self._make_dl1_tree(tmp_path, 10623)
        calls = self._fake_batch(monkeypatch)
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_0.22/dec_347"),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:built"] == 1
        assert calls == [
            ([dl1_file], tmp_path / "out" / "dl2" / "zd_20_30", Path("/models/nsb_tuning_0.22/dec_347"), False)
        ]

    def test_build_resolves_nsb_level_placeholder_per_run(self, tmp_path, monkeypatch):
        dl1_files = self._make_dl1_tree(tmp_path, 10623, 10624)
        calls = self._fake_batch(monkeypatch)
        nsb_by_file = {dl1_files[0]: 0.22, dl1_files[1]: 0.38}
        monkeypatch.setattr(
            producers_module,
            "LSTDL1Event",
            lambda path: SimpleNamespace(nsb_level=nsb_by_file[Path(path)]),
        )
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_$NSB_LEVEL/dec_347"),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert counts["dl2:built"] == 2
        assert {call[2] for call in calls} == {
            Path("/models/nsb_tuning_0.22/dec_347"),
            Path("/models/nsb_tuning_0.38/dec_347"),
        }

    def test_build_skips_existing_dl2_unless_overwrite(self, tmp_path, monkeypatch):
        self._make_dl1_tree(tmp_path, 10623)
        calls = self._fake_batch(monkeypatch)
        dl2_file = tmp_path / "out" / "dl2" / "zd_20_30" / "dl2_LST-1.Run10623.h5"
        dl2_file.parent.mkdir(parents=True)
        dl2_file.touch()
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_0.22/dec_347"),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:existing"] == 1
        assert counts["dl2:built"] == 0
        assert calls == []

        config_overwrite = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_0.22/dec_347", overwrite=True),
        )
        counts = DL2Producer(config_overwrite, tmp_path / "out").produce(make_selection([RUN_10623]))

        assert counts["dl2:built"] == 1
        assert len(calls) == 1
        assert calls[0][3] is True

    def test_build_warns_and_skips_when_no_dl1_directory(self, tmp_path):
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_0.22/dec_347"),
        )
        producer = DL2Producer(config, tmp_path / "out")

        with pytest.warns(UserWarning, match="no DL1 directory"):
            counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {}


DL3_RUN = {"run_number": 14083, "date": 20230821, "zenith_bin": "zd_20_30"}

# Server paths as the finder would return them; only the parsed attributes matter here.
DL3_FULL = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/src_indep/full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"
DL3_POINT = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/src_indep/point/diffuse/gheff0.9_thetacont0.9/irf_interp/dl3_LST-1.Run14083.fits"
DL3_OLD_VERSION = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.10/tailcut1005/nsb_tuning_0.38/0.10.18/src_indep/full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"


def make_dl3_result(newest, all_files):
    return SimpleNamespace(newest_attributes=newest, dl3_files=all_files)


class TestDL3Producer:
    def test_link_places_newest_variants_in_variant_tree(self, tmp_path, monkeypatch):
        newest = [
            make_dl3_attrs(
                DL3_FULL, irf_type="full-enclosure", mc_data_type="ring-wobble", gh_eff=0.7, theta_containment=0.7
            ),
            make_dl3_attrs(DL3_POINT, irf_type="point", mc_data_type="diffuse", gh_eff=0.9, theta_containment=0.9),
        ]
        old = make_dl3_attrs(DL3_OLD_VERSION, lstchain_version="0.10")
        patch_finder(
            monkeypatch,
            {(20230821, 14083, "dl3"): make_dl3_result(newest, [*newest, old])},
        )
        config = DL3ProducerConfig(enabled=True, source="link", link=DL3LinkOptions())
        producer = DL3Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([DL3_RUN]))

        tree = tmp_path / "out" / "dl3" / "zd_20_30"
        assert (
            tree / "full-enclosure" / "ring-wobble" / "gheff0.7_thetacont0.7" / "dl3_LST-1.Run14083.fits"
        ).is_symlink()
        assert (tree / "point" / "diffuse" / "gheff0.9_thetacont0.9" / "dl3_LST-1.Run14083.fits").is_symlink()
        assert counts["dl3:created"] == 2

    @pytest.mark.parametrize("irf_type", ["full-enclosure", "point"])
    def test_link_irf_type_filter_keeps_only_requested_type(self, tmp_path, monkeypatch, irf_type):
        newest = [
            make_dl3_attrs(
                DL3_FULL, irf_type="full-enclosure", mc_data_type="ring-wobble", gh_eff=0.7, theta_containment=0.7
            ),
            make_dl3_attrs(DL3_POINT, irf_type="point", mc_data_type="diffuse", gh_eff=0.9, theta_containment=0.9),
        ]
        patch_finder(monkeypatch, {(20230821, 14083, "dl3"): make_dl3_result(newest, newest)})
        config = DL3ProducerConfig(enabled=True, source="link", link=DL3LinkOptions(irf_type=irf_type))
        producer = DL3Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([DL3_RUN]))

        assert counts["dl3:created"] == 1
        linked = [path for path in (tmp_path / "out" / "dl3" / "zd_20_30").rglob("*.fits")]
        assert len(linked) == 1
        assert linked[0].parent.parent.parent.name == irf_type

    def _make_dl2_tree(self, tmp_path, *run_numbers: int) -> list[Path]:
        files = []
        for run_number in run_numbers:
            dl2_file = tmp_path / "out" / "dl2" / "zd_20_30" / f"dl2_LST-1.Run{run_number:05d}.h5"
            dl2_file.parent.mkdir(parents=True, exist_ok=True)
            dl2_file.touch()
            files.append(dl2_file)
        return files

    def _fake_irf_and_reduction(self, monkeypatch):
        """IRFGenerator/find_irf_nodes/LSTDL2EventTable/DL3Reducer stand-ins."""
        node = SimpleNamespace(relative_path=Path("nsb_0.38/dec_347/theta_20_az_180.irf.fits.gz"))
        made_nodes = []
        reductions = []
        generator_bases = []
        diffuse_modes = []

        def fake_find_irf_nodes(table, *, diffuse=True):
            diffuse_modes.append(diffuse)
            return [node]

        class FakeIRFGenerator:
            def __init__(self, base_path, irf_config):
                self.base_path = Path(base_path)
                generator_bases.append(self.base_path)

            def make_irf_node(self, node):
                made_nodes.append(node)

        class FakeDL3Reducer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reduce(self, dl2_files, output_dir):
                dl2_files = list(dl2_files)
                reductions.append((dl2_files, Path(output_dir)))
                return [dl2_file.with_suffix(".fits") for dl2_file in dl2_files]

        monkeypatch.setattr(producers_module, "LSTDL2EventTable", lambda path: object())
        monkeypatch.setattr(producers_module, "find_irf_nodes", fake_find_irf_nodes)
        monkeypatch.setattr(producers_module, "IRFGenerator", FakeIRFGenerator)
        monkeypatch.setattr(producers_module, "DL3Reducer", FakeDL3Reducer)
        return made_nodes, reductions, generator_bases, diffuse_modes

    @staticmethod
    def _build_config(**overrides):
        options = {
            "output_directory": None,
            "irf_directory": None,
            "event_filter": EventFilter.from_cuts(intensity_cut=80, leakage_cut=1.0, gh_cut=0.7),
        }
        options.update(overrides)
        return DL3ProducerConfig(enabled=True, source="build", build=DL3BuildOptions(**options))

    def test_build_generates_irfs_and_reduces_per_bin(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624)
        made_nodes, reductions, generator_bases, diffuse_modes = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig(name="Crab Nebula", ra=83.632, dec=22.0145),
        )

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        # one node although two DL2 files were scanned (deduplicated by relative path)
        assert len(made_nodes) == 1
        assert counts["dl3:irf_nodes"] == 1
        assert counts["dl3:reduced"] == 2
        assert counts["dl3:failed"] == 0
        assert generator_bases == [tmp_path / "out" / "irfs"]
        assert reductions == [(dl2_files, tmp_path / "out" / "dl3" / "zd_20_30")]
        assert diffuse_modes == [True, True]

    def test_build_uses_configured_irf_directory(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, reductions, generator_bases, _ = self._fake_irf_and_reduction(monkeypatch)
        config = self._build_config(irf_directory=tmp_path / "my_irfs")
        producer = DL3Producer(config, tmp_path / "out", source=SourceConfig(name="Crab", ra=83.6, dec=22.0))

        producer.produce(make_selection([RUN_10623]))

        assert generator_bases == [tmp_path / "my_irfs"]
        assert len(reductions) == 1

    def test_build_configures_point_like_irf_generation(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, _, _, diffuse_modes = self._fake_irf_and_reduction(monkeypatch)
        irf_configs = []
        real_irf_config = producers_module.IRFConfig

        def capture_irf_config(*args, **kwargs):
            config = real_irf_config(*args, **kwargs)
            irf_configs.append(config)
            return config

        monkeypatch.setattr(producers_module, "IRFConfig", capture_irf_config)
        producer = DL3Producer(
            self._build_config(point_like=True),
            tmp_path / "out",
            source=SourceConfig(name="Crab", ra=83.6, dec=22.0),
        )

        producer.produce(make_selection([RUN_10623]))

        assert [config.point_like for config in irf_configs] == [True]
        assert diffuse_modes == [False]

    def test_build_requires_source(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(self._build_config(), tmp_path / "out")

        with pytest.raises(ValueError, match="requires the analysis source"):
            producer.produce(make_selection([RUN_10623]))

    def test_build_warns_and_skips_when_no_dl2_directory(self, tmp_path, monkeypatch):
        self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig(name="Crab", ra=83.6, dec=22.0),
        )

        with pytest.warns(UserWarning, match="no DL2 directory"):
            counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {}
