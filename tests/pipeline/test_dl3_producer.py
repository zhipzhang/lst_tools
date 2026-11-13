import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal

import pandas as pd
import pytest

import lst_tools.pipeline.dl3_build as dl3_module
import lst_tools.pipeline.producers as producers_module
from lst_tools.finder import MC_DL2_PATH, DL3Metadata, MCDL2Metadata
from lst_tools.irf import IRFConfig
from lst_tools.pipeline import (
    DL3BuildOptions,
    DL3LinkOptions,
    DL3Producer,
    DL3ProducerConfig,
    DL3ReductionOptions,
    RunSelection,
    SlurmConfig,
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


def make_dl3_attrs(
    path: str,
    irf_type: Literal["full-enclosure", "point"] = "full-enclosure",
    mc_data_type: Literal["ring-wobble", "diffuse"] = "ring-wobble",
    gh_eff: float = 0.7,
    theta_containment: float = 0.7,
    lstchain_version: str = "0.11",
) -> DL3Metadata:
    return DL3Metadata(
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
    def fake_find(date, run_number, level):
        return results_by_key[(date, run_number, level)]

    monkeypatch.setattr(producers_module, "find_la_palma_run", fake_find)


RUN_10623 = {"runnumber": 10623, "date": 20220426}
RUN_10624 = {"runnumber": 10624, "date": 20220426}
DL3_RUN = {"runnumber": 14083, "date": 20230821}

# Server paths as the finder would return them; only the parsed attributes matter here.
DL3_FULL = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/src_indep/full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"
DL3_POINT = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/src_indep/point/diffuse/gheff0.9_thetacont0.9/irf_interp/dl3_LST-1.Run14083.fits"
DL3_OLD_VERSION = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.10/tailcut1005/nsb_tuning_0.38/0.10.18/src_indep/full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"


def make_dl3_result(newest, all_files):
    return SimpleNamespace(newest_metadata=newest, dl3_files=all_files)


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
            MCDL2Metadata(Path("/mc/diffuse.h5"), "GammaDiffuse", 20, 180, 0.38, 347),
            MCDL2Metadata(Path("/mc/gamma.h5"), "Gamma", 20, 180, 0.38, 347),
            MCDL2Metadata(Path("/mc/other_dec.h5"), "GammaDiffuse", 20, 180, 0.38, 999),
        ]
        made_samples = []
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

            def generate_all(self, samples):
                made_samples.extend(samples)

        class FakeDL3Reducer:
            def __init__(self, **kwargs):
                self.irfs_root = Path(kwargs["irfs_root"])
                self.output_dir = Path(kwargs["output_dir"])
                self.irf_config = kwargs["irf_config"]

            def reduce(self, dl2_files):
                assert made_samples
                dl2_files = list(dl2_files)
                reductions.append((dl2_files, self.output_dir))
                return [dl2_file.with_suffix(".fits") for dl2_file in dl2_files]

        table = SimpleNamespace(nsb_level=0.38, dec_line=347)
        ImmediateExecutor.workers = []
        monkeypatch.setattr(dl3_module, "ProcessPoolExecutor", ImmediateExecutor)
        monkeypatch.setattr(dl3_module, "LSTObservedDL2Table", lambda path: table)
        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", fake_find_mc_dl2_files)
        monkeypatch.setattr(dl3_module, "IRFGenerator", FakeIRFGenerator)
        monkeypatch.setattr(dl3_module, "DL3Reducer", FakeDL3Reducer)
        return made_samples, reductions, generator_bases, campaigns

    @staticmethod
    def _build_config(**overrides):
        point_like = overrides.pop("point_like", False)
        irf_config = overrides.pop("irf_config", None)
        if irf_config is None:
            irf_config = IRFConfig.from_cuts(
                intensity_cut=80,
                leakage_cut=1.0,
                gh_cut=0.7,
                point_like=point_like,
                theta_cut=0.2 if point_like else None,
            )
        reduction = DL3ReductionOptions(name="standard", irf_config=irf_config)
        options = {
            "output_directory": None,
            "irf_directory": None,
            "reductions": overrides.pop("reductions", (reduction,)),
        }
        options.update(overrides)
        return DL3ProducerConfig(enabled=True, source="build", build=DL3BuildOptions(**options))

    def test_build_generates_irfs_and_reduces_all_selected_runs(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624)
        made_samples, reductions, generator_bases, campaigns = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab Nebula", 83.632, 22.0145)
        )

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert len(made_samples) == 1
        assert made_samples[0].path == Path("/mc/diffuse.h5")
        assert made_samples[0].mc_type == "diffuse"
        assert counts == {"dl3:irf_files": 1, "dl3:reduced": 2, "dl3:failed": 0}
        assert generator_bases == [tmp_path / "out" / "irfs" / "standard"]
        assert reductions == [(dl2_files, tmp_path / "out" / "dl3" / "standard")]
        assert campaigns == [MC_DL2_PATH / "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.38"]

    @pytest.mark.parametrize("point_like", [False, True])
    def test_build_selects_mc_sample_for_irf_type(self, tmp_path, monkeypatch, point_like):
        self._make_dl2_tree(tmp_path, 10623)
        made_samples, _, _, _ = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(point_like=point_like), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0)
        )

        producer.produce(make_selection([RUN_10623]))

        assert made_samples[0].path == Path("/mc/gamma.h5" if point_like else "/mc/diffuse.h5")
        assert made_samples[0].mc_type == ("point-source" if point_like else "diffuse")

    def test_build_prepares_each_mc_group_then_reduces(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624)
        made_samples, _, _, _ = self._fake_irf_and_reduction(monkeypatch)
        nsb_by_file = dict(zip(dl2_files, (0.22, 0.38), strict=True))
        monkeypatch.setattr(
            dl3_module,
            "LSTObservedDL2Table",
            lambda path: SimpleNamespace(nsb_level=nsb_by_file[Path(path)], dec_line=347),
        )

        def find_mc(directory):
            nsb = float(str(directory).rsplit("_", 1)[1])
            return [MCDL2Metadata(Path("/mc/diffuse.h5"), "GammaDiffuse", 20, 180, nsb, 347)]

        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", find_mc)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))

        counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        assert {sample.nsb_level for sample in made_samples} == {0.22, 0.38}
        assert counts["dl3:irf_files"] == 2
        assert ImmediateExecutor.workers == [2, 1]

    def test_build_isolates_multiple_reductions(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623)
        made_samples, reductions, generator_bases, _ = self._fake_irf_and_reduction(monkeypatch)
        reduction_options = tuple(
            DL3ReductionOptions(name, IRFConfig.from_cuts(intensity, 1.0, gh_cut=0.7))
            for name, intensity in (("standard", 80), ("quality_tight", 150))
        )
        producer = DL3Producer(
            self._build_config(reductions=reduction_options), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0)
        )
        counts = producer.produce(make_selection([RUN_10623]))

        assert [sample.path for sample in made_samples] == [Path("/mc/diffuse.h5")] * 2
        assert generator_bases == [tmp_path / "out" / "irfs" / name for name in ("standard", "quality_tight")]
        assert reductions == [(dl2_files, tmp_path / "out" / "dl3" / name) for name in ("standard", "quality_tight")]
        assert counts["dl3:irf_files"] == 2
        assert counts["dl3:reduced"] == 2
        assert ImmediateExecutor.workers == [1, 1, 1, 1]

    def test_build_uses_configured_directories(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, reductions, generator_bases, campaigns = self._fake_irf_and_reduction(monkeypatch)
        config = self._build_config(
            irf_directory=tmp_path / "my_irfs",
            output_directory=tmp_path / "my_dl3",
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
    def test_missing_selected_dl2_warns_and_reduces_available_runs(self, tmp_path, monkeypatch, slurm):
        (dl2_file,) = self._make_dl2_tree(tmp_path, 10623)
        made_samples, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        if slurm:
            import simple_slurm

            monkeypatch.setattr(simple_slurm.Slurm, "sbatch", lambda slurm_job: 12345)
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=SlurmConfig(enabled=slurm),
        )
        with pytest.warns(UserWarning, match="missing DL2: 10624"):
            counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        if slurm:
            assert counts == {"dl3:slurm_jobs": 1}
        else:
            assert reductions == [([dl2_file], tmp_path / "out" / "dl3" / "standard")]
            assert len(made_samples) == 1
            assert len(campaigns) == 1

    @pytest.mark.parametrize("slurm", [False, True])
    def test_missing_dl2_directory_warns_and_does_no_work(self, tmp_path, monkeypatch, slurm):
        made_samples, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=SlurmConfig(enabled=slurm),
        )
        with pytest.warns(UserWarning, match="no DL2 directory") as warnings:
            counts = producer.produce(make_selection([RUN_10623, RUN_10624]))

        message = str(warnings[0].message)
        assert "10623" in message and "10624" in message
        assert counts == {}
        assert not made_samples and not reductions and not campaigns
        assert not (tmp_path / "out" / "slurm_jobs").exists()

    def test_unselected_dl2_files_are_never_read_or_reduced(self, tmp_path, monkeypatch):
        files = self._make_dl2_tree(tmp_path, 10623, 10624, 99999)
        files[0].with_name("dl2_unrelated.h5").touch()
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        reads = []

        def read_table(path):
            reads.append(path)
            return SimpleNamespace(nsb_level=0.38, dec_line=347)

        monkeypatch.setattr(dl3_module, "LSTObservedDL2Table", read_table)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        producer.produce(make_selection([RUN_10623]))

        assert reads == [files[0]]
        assert reductions[0][0] == [files[0]]

    def test_ambiguous_selected_dl2_warns_and_is_skipped(self, tmp_path, monkeypatch):
        (file,) = self._make_dl2_tree(tmp_path, 10623)
        file.with_name("dl2_LST-1.Run10623.extra.h5").touch()
        made_samples, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.warns(UserWarning, match="ambiguous DL2: 10623"):
            counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {}
        assert not made_samples and not reductions and not campaigns

    @pytest.mark.parametrize("slurm", [False, True])
    def test_empty_selection_does_no_work(self, tmp_path, monkeypatch, slurm):
        self._make_dl2_tree(tmp_path, 10623)
        made_samples, reductions, _, campaigns = self._fake_irf_and_reduction(monkeypatch)
        selection = make_selection([RUN_10623])
        selection.selected = selection.selected.iloc[:0]
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=SlurmConfig(enabled=slurm),
        )
        assert producer.produce(selection) == {}
        assert not made_samples and not reductions and not campaigns
        assert not (tmp_path / "out" / "slurm_jobs").exists()

    def test_missing_mc_samples_stop_before_reduction(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        made_samples, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        monkeypatch.setattr(dl3_module, "find_mc_dl2_files", lambda path: [])
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(FileNotFoundError, match="No GammaDiffuse MC samples"):
            producer.produce(make_selection([RUN_10623]))
        assert not made_samples and not reductions

    def test_irf_failure_prevents_reduction(self, tmp_path, monkeypatch):
        self._make_dl2_tree(tmp_path, 10623)
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)

        class FailingIRFGenerator:
            def __init__(self, base_path, irf_config):
                pass

            def generate_all(self, samples):
                raise RuntimeError("IRF generation failed")

        monkeypatch.setattr(dl3_module, "IRFGenerator", FailingIRFGenerator)
        producer = DL3Producer(self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0))
        with pytest.raises(RuntimeError, match="IRF generation failed"):
            producer.produce(make_selection([RUN_10623]))
        assert not reductions

    def test_build_chunks_dl2_files_by_runs_per_worker(self, tmp_path, monkeypatch):
        dl2_files = self._make_dl2_tree(tmp_path, 10623, 10624, 10625)
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        slurm_config = SlurmConfig(
            enabled=False,
            dl3=SlurmDL3Config(runs_per_worker=2),
        )
        producer = DL3Producer(
            self._build_config(),
            tmp_path / "out",
            source=SourceConfig("Crab", 83.6, 22.0),
            slurm_config=slurm_config,
        )
        selection = make_selection(
            [
                {"runnumber": 10623, "date": 20220426},
                {"runnumber": 10624, "date": 20220426},
                {"runnumber": 10625, "date": 20220426},
            ]
        )

        counts = producer.produce(selection)

        assert counts == {"dl3:irf_files": 1, "dl3:reduced": 3, "dl3:failed": 0}
        assert reductions == [
            (dl2_files[:2], tmp_path / "out" / "dl3" / "standard"),
            (dl2_files[2:], tmp_path / "out" / "dl3" / "standard"),
        ]
        assert ImmediateExecutor.workers == [1, 2]

    def test_slurm_writes_irf_config_and_worker_executes_command(self, tmp_path, monkeypatch):
        import simple_slurm

        from lst_tools.scripts import dl2_to_dl3

        files = self._make_dl2_tree(tmp_path, 10623, 10624)
        _, reductions, _, _ = self._fake_irf_and_reduction(monkeypatch)
        slurm_config = SlurmConfig(
            enabled=True,
            partition="long",
            dl3=SlurmDL3Config(runs_per_worker=3, cpus_per_task=1, mem="32GB", time="06:00:00"),
        )
        producer = DL3Producer(
            self._build_config(), tmp_path / "out", source=SourceConfig("Crab", 83.6, 22.0), slurm_config=slurm_config
        )
        scripts = []
        monkeypatch.setattr(simple_slurm.Slurm, "sbatch", lambda slurm: scripts.append(slurm.script()) or 12345)
        counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {"dl3:slurm_jobs": 1}
        irf_config_file = tmp_path / "out" / "irfs" / "standard" / "irf_config.json"
        assert irf_config_file.is_file()
        assert "--irf-config" in scripts[0] and str(irf_config_file) in scripts[0]
        assert "--max-workers 1" in scripts[0]
        assert "--runs-per-worker 3" in scripts[0]
        assert "#SBATCH --cpus-per-task       1" in scripts[0]
        assert "#SBATCH --mem                 32GB" in scripts[0]

        command_line = next(line for line in scripts[0].splitlines() if line.startswith("lst-dl2-to-dl3"))
        worker_argv = shlex.split(command_line)[1:]
        assert dl2_to_dl3.main(worker_argv)["dl3:reduced"] == 1
        assert reductions[0][0] == [files[0]]
        assert ImmediateExecutor.workers == [1, 1]
        files[0].unlink()
        with pytest.raises(FileNotFoundError, match="selected DL2 files no longer exist"):
            dl2_to_dl3.main(worker_argv)
