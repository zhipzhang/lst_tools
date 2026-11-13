import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest

import lst_tools.pipeline.dl3 as dl3_module
import lst_tools.pipeline.dl3_execution as execution_module
import lst_tools.pipeline.producers as producers_module
from lst_tools.event_filter import EventFilter
from lst_tools.finder import MC_DL2_PATH, DL3Attributes, DLAttributes, MCDL2Attribute
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
    DL3ReductionOptions,
    RunSelection,
    SlurmConfig,
    SlurmDL2Config,
    SlurmDL3Config,
    SourceConfig,
)


def make_selection(rows) -> RunSelection:
    return RunSelection(selected=pd.DataFrame(rows))


class ImmediateExecutor:
    workers: ClassVar[list[int]] = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        type(self).workers.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map(self, worker, tasks):
        return [worker(task) for task in tasks]


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


RUN_10623 = {"runnumber": 10623, "date": 20220426}
RUN_10624 = {"runnumber": 10624, "date": 20220426}


class TestDL1Producer:
    def test_link_places_newest_files_in_flat_directory(self, tmp_path, monkeypatch):
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

        assert (tmp_path / "out" / "dl1" / source_1.name).is_symlink()
        assert (tmp_path / "out" / "dl1" / source_2.name).is_symlink()
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
        taken = output_root / "dl1" / source.name
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

        assert (tmp_path / "custom_dl1" / source.name).is_symlink()
        assert not (tmp_path / "out" / "dl1").exists()

    def test_disabled_producer_does_nothing(self, tmp_path, monkeypatch):
        def forbidden_find(date, run_number, level):  # pragma: no cover - must not be called
            raise AssertionError("a disabled producer must not search the server")

        monkeypatch.setattr(producers_module, "find_la_palma_files", forbidden_find)
        producer = DL1Producer(DL1ProducerConfig(enabled=False), tmp_path / "out")

        assert producer.produce(make_selection([RUN_10623])) == {}


class TestDL2Producer:
    # TEST: Local and SLURM cases should share one pending-plan test suite, retaining only
    # execution-specific assertions here (local reconstruction versus job command submission).
    def test_link_uses_same_tree_as_dl1(self, tmp_path, monkeypatch):
        source = tmp_path / "server" / "dl2_LST-1.Run10623.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        patch_finder(monkeypatch, {(20220426, 10623, "dl2"): SimpleNamespace(newest_attributes=make_dl_attrs(source))})
        producer = DL2Producer(DL2ProducerConfig(enabled=True, source="link"), tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert (tmp_path / "out" / "dl2" / source.name).is_symlink()
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
            dl1_file = tmp_path / "out" / "dl1" / f"dl1_LST-1.Run{run_number:05d}.h5"
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

        monkeypatch.setattr("lst_tools.dl2.reconstruction.dl1_to_dl2_batch", fake_batch)
        return calls

    # TEST: Add a tool-completes-without-output case to verify warning/failed accounting,
    # and a no-pending SLURM case to verify that no empty job is submitted.
    def test_build_reconstructs_only_selected_pending_dl1(self, tmp_path, monkeypatch):
        dl1_file, _ = self._make_dl1_tree(tmp_path, 10623, 99999)
        calls = self._fake_batch(monkeypatch)
        model_directory = tmp_path / "models"
        model_directory.mkdir()
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(model_directory)),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:built"] == 1
        assert calls == [([dl1_file], tmp_path / "out" / "dl2", model_directory, False)]

    def test_build_resolves_nsb_level_placeholder_per_run(self, tmp_path, monkeypatch):
        dl1_files = self._make_dl1_tree(tmp_path, 10623, 10624)
        calls = self._fake_batch(monkeypatch)
        nsb_by_file = {dl1_files[0]: 0.22, dl1_files[1]: 0.38}
        monkeypatch.setattr(
            "lst_tools.dl2.reconstruction.resolve_dl1_nsb_level",
            lambda path: nsb_by_file[Path(path)],
        )
        model_directories = {
            tmp_path / "models" / "nsb_tuning_0.22" / "dec_347",
            tmp_path / "models" / "nsb_tuning_0.38" / "dec_347",
        }
        for model_directory in model_directories:
            model_directory.mkdir(parents=True)
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(tmp_path / "models" / "nsb_tuning_$NSB_LEVEL" / "dec_347")),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert counts["dl2:built"] == 2
        assert {call[2] for call in calls} == model_directories

    def test_build_skips_existing_dl2_unless_overwrite(self, tmp_path, monkeypatch):
        self._make_dl1_tree(tmp_path, 10623)
        calls = self._fake_batch(monkeypatch)
        model_directory = tmp_path / "models"
        model_directory.mkdir()
        dl2_file = tmp_path / "out" / "dl2" / "dl2_LST-1.Run10623.h5"
        dl2_file.parent.mkdir(parents=True)
        dl2_file.touch()
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(model_directory)),
        )
        producer = DL2Producer(config, tmp_path / "out")

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:existing"] == 1
        assert counts["dl2:built"] == 0
        assert calls == []

        config_overwrite = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(model_directory), overwrite=True),
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

    def test_slurm_build_submits_run_chunks_without_nsb_lookup(self, tmp_path, monkeypatch):
        self._make_dl1_tree(tmp_path, 10001, 10002, 10003)
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_$NSB_LEVEL/dec_347"),
        )
        slurm_cfg = SlurmConfig(
            enabled=True,
            partition="short",
            account="aswg",
            env_setup="conda activate lst-env",
            dl2=SlurmDL2Config(runs_per_job=2),
        )
        producer = DL2Producer(config, tmp_path / "out", slurm_config=slurm_cfg)

        def unexpected_nsb_lookup(path):
            raise AssertionError(f"SLURM submission looked up NSB for {path}")

        monkeypatch.setattr("lst_tools.dl2.reconstruction.resolve_dl1_nsb_level", unexpected_nsb_lookup)
        submitted_scripts = []
        counter = [0]

        def fake_sbatch(self_slurm, *args, **kwargs):
            submitted_scripts.append(self_slurm.script())
            counter[0] += 1
            return counter[0]

        import simple_slurm

        monkeypatch.setattr(simple_slurm.Slurm, "sbatch", fake_sbatch)

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:slurm_jobs"] == 2
        assert counts["dl2:slurm_runs"] == 3
        assert len(submitted_scripts) == 2
        assert "#SBATCH --account             aswg" in submitted_scripts[0]
        assert "#SBATCH --partition           short" in submitted_scripts[0]
        assert "conda activate lst-env" in submitted_scripts[0]
        assert f"{DL2Producer.worker_command} --input" in submitted_scripts[0]
        assert r"--model-dir '/models/nsb_tuning_\$NSB_LEVEL/dec_347'" in submitted_scripts[0]

    def test_slurm_build_skips_existing_unless_overwrite(self, tmp_path, monkeypatch):
        self._make_dl1_tree(tmp_path, 10001, 10002)
        dl2_file = tmp_path / "out" / "dl2" / "dl2_LST-1.Run10001.h5"
        dl2_file.parent.mkdir(parents=True, exist_ok=True)
        dl2_file.touch()

        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models/nsb_tuning_0.22/dec_347"),
        )
        slurm_cfg = SlurmConfig(enabled=True, partition="short")
        producer = DL2Producer(config, tmp_path / "out", slurm_config=slurm_cfg)

        import simple_slurm

        monkeypatch.setattr(simple_slurm.Slurm, "sbatch", lambda self_slurm: 1)

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:existing"] == 1
        assert counts["dl2:slurm_jobs"] == 1
        assert counts["dl2:slurm_runs"] == 1

    def test_produce_uses_local_build_when_slurm_disabled(self, tmp_path, monkeypatch):
        self._make_dl1_tree(tmp_path, 10001)
        calls = self._fake_batch(monkeypatch)
        model_directory = tmp_path / "models"
        model_directory.mkdir()
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(model_directory)),
        )
        slurm_cfg = SlurmConfig(enabled=False)
        producer = DL2Producer(config, tmp_path / "out", slurm_config=slurm_cfg)

        counts = producer.produce(make_selection([{"runnumber": 10001, "date": 20220426}]))

        assert counts["dl2:built"] == 1
        assert len(calls) == 1

    def test_default_slurm_build_raises_not_implemented(self, tmp_path):
        config = DL1ProducerConfig(enabled=True, source="build")  # type: ignore[arg-type]
        producer = DL1Producer(config, tmp_path / "out")
        with pytest.raises(NotImplementedError, match="SLURM building is not supported yet"):
            producer.slurm_build(make_selection([RUN_10623]), SlurmConfig(enabled=True))



DL3_RUN = {"runnumber": 14083, "date": 20230821}

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

        tree = tmp_path / "out" / "dl3"
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
        linked = [path for path in (tmp_path / "out" / "dl3").rglob("*.fits")]
        assert len(linked) == 1
        assert linked[0].parent.parent.parent.name == irf_type

    def _make_dl2_tree(self, tmp_path, *run_numbers: int) -> list[Path]:
        files = []
        for run_number in run_numbers:
            dl2_file = tmp_path / "out" / "dl2" / f"dl2_LST-1.Run{run_number:05d}.h5"
            dl2_file.parent.mkdir(parents=True, exist_ok=True)
            dl2_file.touch()
            files.append(dl2_file)
        return files

    def _fake_irf_and_reduction(self, monkeypatch):
        attributes = [
            MCDL2Attribute(Path("/mc/diffuse.h5"), "GammaDiffuse", 20, 180, 0.38, 347),
            MCDL2Attribute(Path("/mc/gamma.h5"), "Gamma", 20, 180, 0.38, 347),
            MCDL2Attribute(Path("/mc/other_dec.h5"), "GammaDiffuse", 20, 180, 0.38, 999),
        ]
        made_nodes = []
        reductions = []
        generator_bases = []
        campaigns = []

        def fake_find_mc_dl2_files(directory):
            campaigns.append(Path(directory))
            return attributes

        class FakeIRFGenerator:
            def __init__(self, base_path, irf_config):
                self.base_path = Path(base_path)
                generator_bases.append(self.base_path)

            def make_irf_nodes(self, nodes):
                made_nodes.extend(nodes)

        class FakeDL3Reducer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reduce(self, dl2_files, output_dir):
                assert made_nodes
                dl2_files = list(dl2_files)
                reductions.append((dl2_files, Path(output_dir)))
                return [dl2_file.with_suffix(".fits") for dl2_file in dl2_files]

        table = SimpleNamespace(nsb_level=0.38, dec_line=347)
        ImmediateExecutor.workers = []
        monkeypatch.setattr(execution_module, "ProcessPoolExecutor", ImmediateExecutor)
        monkeypatch.setattr(dl3_module, "LSTDL2EventTable", lambda path: table)
        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", fake_find_mc_dl2_files)
        monkeypatch.setattr(dl3_module, "IRFGenerator", FakeIRFGenerator)
        monkeypatch.setattr(dl3_module, "DL3Reducer", FakeDL3Reducer)
        return made_nodes, reductions, generator_bases, campaigns

    @staticmethod
    def _build_config(**overrides):
        reduction = DL3ReductionOptions(
            name="standard",
            event_filter=overrides.pop(
                "event_filter", EventFilter.from_cuts(intensity_cut=80, leakage_cut=1.0, gh_cut=0.7)
            ),
            point_like=overrides.pop("point_like", False),
        )
        options = {
            "output_directory": None,
            "irf_directory": None,
            "reductions": overrides.pop("reductions", (reduction,)),
        }
        options.update(overrides)
        return DL3ProducerConfig(enabled=True, source="build", build=DL3BuildOptions(**options))

    def test_build_generates_irfs_and_reduces_all_selected_runs(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624)
        made_nodes, reductions, generator_bases, campaigns = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab Nebula", 83.632, 22.0145)
        )

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert len(made_nodes) == 1
        assert made_nodes[0].dl2_path == Path("/mc/diffuse.h5")
        assert made_nodes[0].irf_type == "full-enclosure"
        assert counts == {"dl3:irf_nodes": 1, "dl3:reduced": 2, "dl3:failed": 0}
        assert generator_bases == [tmp_path / "out" / "irfs" / "standard"]
        assert reductions == [(dl2_files, tmp_path / "out" / "dl3" / "standard")]
        assert campaigns == [MC_DL2_PATH / "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.38"]

    @pytest.mark.parametrize("point_like", [False, True])
    def test_build_applies_reduction_to_irf_nodes(self, tmp_path, monkeypatch, point_like):
        self._make_dl2_tree(tmp_path, 10623)
        made_nodes, _, _, _ = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(point_like=point_like), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0)
        )

        producer.produce(make_selection([RUN_10623]))

        assert made_nodes[0].gh_type == "cut"
        assert made_nodes[0].gh_value == pytest.approx(0.7)
        assert made_nodes[0].dl2_path == Path("/mc/gamma.h5" if point_like else "/mc/diffuse.h5")
        assert made_nodes[0].irf_type == ("point" if point_like else "full-enclosure")

    def test_build_prepares_each_mc_group_then_reduces(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624)
        made_nodes, _, _, _ = self._fake_irf_and_reduction(monkeypatch)
        nsb_by_file = dict(zip(dl2_files, (0.22, 0.38), strict=True))
        monkeypatch.setattr(
            dl3_module, "LSTDL2EventTable",
            lambda path: SimpleNamespace(nsb_level=nsb_by_file[Path(path)], dec_line=347),
        )

        def find_mc(directory):
            nsb = float(str(directory).rsplit("_", 1)[1])
            return [MCDL2Attribute(Path("/mc/diffuse.h5"), "GammaDiffuse", 20, 180, nsb, 347)]

        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", find_mc)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert {node.nsb_level for node in made_nodes} == {0.22, 0.38}
        assert counts["dl3:irf_nodes"] == 2
        assert ImmediateExecutor.workers == [2, 1]

    def test_build_isolates_reductions_and_shares_discovery(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623)
        made_nodes, reductions, generator_bases, campaigns = self._fake_irf_and_reduction(monkeypatch)
        reduction_options = tuple(
            DL3ReductionOptions(name, EventFilter.from_cuts(intensity, 1.0, gh_cut=0.7))
            for name, intensity in (("standard", 80), ("quality_tight", 150))
        )
        producer = DL3Producer(
            self._build_config(reductions=reduction_options), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0)
        )
        provenance_reads = []

        def read_table(path):
            provenance_reads.append(path)
            return SimpleNamespace(nsb_level=0.38, dec_line=347)

        monkeypatch.setattr(dl3_module, "LSTDL2EventTable", read_table)
        counts = producer.produce(make_selection([RUN_10623]))

        assert [node.cut_label for node in made_nodes] == ["ghcut0.7", "ghcut0.7"]
        assert generator_bases == [tmp_path / "out" / "irfs" / name for name in ("standard", "quality_tight")]
        assert reductions == [
            (dl2_files, tmp_path / "out" / "dl3" / name) for name in ("standard", "quality_tight")
        ]
        assert counts["dl3:irf_nodes"] == 2
        assert counts["dl3:reduced"] == 2
        assert len(campaigns) == 1
        assert provenance_reads == dl2_files
        assert ImmediateExecutor.workers == [2, 2]

    def test_build_uses_configured_directories(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, reductions, generator_bases, campaigns = self._fake_irf_and_reduction(monkeypatch)
        config = self._build_config(
            irf_directory=tmp_path / "my_irfs", output_directory=tmp_path / "my_dl3",
            mc_dl2_directory=str(tmp_path / "mc_$NSB_LEVEL"),
        )
        producer = DL3Producer(config, tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))

        producer.produce(make_selection([RUN_10623]))

        assert generator_bases == [tmp_path / "my_irfs" / "standard"]
        assert reductions[0][1] == tmp_path / "my_dl3" / "standard"
        assert campaigns == [tmp_path / "mc_0.38"]

    def test_build_requires_source(self, tmp_path):
        producer = DL3Producer(self._build_config(), tmp_path / "out")
        with pytest.raises(ValueError, match="requires the analysis source"):
            producer.produce(make_selection([RUN_10623]))

    @pytest.mark.parametrize("slurm", [False, True])
    def test_missing_selected_dl2_stops_before_work(self, tmp_path, monkeypatch, slurm):
        self._make_dl2_tree(tmp_path, 10623)
        made_nodes, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=SlurmConfig(enabled=slurm),
        )
        with pytest.raises(FileNotFoundError, match="missing DL2 files for selected runs") as error:
            producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert "10624" in str(error.value)
        assert str(tmp_path / "out" / "dl2") in str(error.value)
        assert not made_nodes and not reductions and not campaigns
        assert not (tmp_path / "out" / "slurm_jobs").exists()

    def test_missing_dl2_directory_lists_all_selected_runs(self, tmp_path):
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(FileNotFoundError) as error:
            producer.produce(make_selection([RUN_10623, RUN_10624]))
        assert "10623" in str(error.value) and "10624" in str(error.value)

    def test_unselected_dl2_files_are_never_read_or_reduced(self, tmp_path, monkeypatch):
        files = self._make_dl2_tree(tmp_path, 10623, 10624, 99999)
        files[0].with_name("dl2_unrelated.h5").touch()
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        reads = []

        def read_table(path):
            reads.append(path)
            return SimpleNamespace(nsb_level=0.38, dec_line=347)

        monkeypatch.setattr(dl3_module, "LSTDL2EventTable", read_table)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        producer.produce(make_selection([RUN_10623]))

        assert reads == [files[0]]
        assert reductions[0][0] == [files[0]]

    def test_ambiguous_selected_dl2_is_rejected(self, tmp_path):
        (file,) = self._make_dl2_tree(tmp_path, 10623)
        file.with_name("dl2_LST-1.Run10623.extra.h5").touch()
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(ValueError, match="multiple DL2 files for selected run 10623"):
            producer.produce(make_selection([RUN_10623]))

    @pytest.mark.parametrize("slurm", [False, True])
    def test_empty_selection_does_no_work(self, tmp_path, monkeypatch, slurm):
        self._make_dl2_tree(tmp_path, 10623)
        made_nodes, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        selection = make_selection([RUN_10623])
        selection.selected = selection.selected.iloc[:0]
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=SlurmConfig(enabled=slurm),
        )
        assert producer.produce(selection) == {}
        assert not made_nodes and not reductions and not campaigns
        assert not (tmp_path / "out" / "slurm_jobs").exists()

    def test_missing_mc_nodes_stop_before_reduction(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        made_nodes, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", lambda path: [])
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(FileNotFoundError, match="No GammaDiffuse MC nodes"):
            producer.produce(make_selection([RUN_10623]))
        assert not made_nodes and not reductions

    def test_irf_failure_prevents_reduction(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)

        def fail(self):
            raise RuntimeError("IRF generation failed")

        monkeypatch.setattr(dl3_module.IRFSet, "prepare", fail)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(RuntimeError, match="IRF generation failed"):
            producer.produce(make_selection([RUN_10623]))
        assert not reductions

    def test_slurm_serializes_selection_and_worker_executes_it(self, tmp_path, monkeypatch):
        import simple_slurm

        from lst_tools.scripts.dl3_build import build_dl3

        files = self._make_dl2_tree(tmp_path, 10623, 10624)
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        slurm_config = SlurmConfig(
            enabled=True, partition="long", dl3=SlurmDL3Config(cpus_per_task=1, mem="32GB", time="06:00:00")
        )
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0), slurm_config=slurm_config
        )
        scripts = []
        monkeypatch.setattr(simple_slurm.Slurm, "sbatch", lambda slurm: scripts.append(slurm.script()) or 12345)
        counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {"dl3:slurm_jobs": 1, "dl3:slurm_reductions": 1}
        (build_pickle,) = (tmp_path / "out" / "slurm_jobs").glob("dl3_build_*.pkl")
        with build_pickle.open("rb") as file_handle:
            restored = pickle.load(file_handle)
        assert isinstance(restored, dl3_module.DL3Build)
        assert [observation.path for observation in restored.observations] == [files[0]]
        assert [reduction.name for reduction in restored.reductions] == ["standard"]
        assert "--build-pickle" in scripts[0] and str(build_pickle) in scripts[0]
        assert "--max-workers 1" in scripts[0]
        assert "#SBATCH --cpus-per-task       1" in scripts[0]
        assert "#SBATCH --mem                 32GB" in scripts[0]

        assert build_dl3(build_pickle, max_workers=1)["dl3:reduced"] == 1
        assert reductions[0][0] == [files[0]]
        assert ImmediateExecutor.workers == [1, 1]
        files[0].unlink()
        with pytest.raises(FileNotFoundError, match="selected DL2 files no longer exist"):
            build_dl3(build_pickle, max_workers=1)
