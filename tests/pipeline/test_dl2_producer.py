from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import lst_tools.pipeline.producers as producers_module
from lst_tools.finder import DLAttributes
from lst_tools.pipeline import (
    DL2BuildOptions,
    DL2LinkOptions,
    DL2Producer,
    DL2ProducerConfig,
    RunSelection,
    SlurmConfig,
)


def make_selection(rows) -> RunSelection:
    return RunSelection(selected=pd.DataFrame(rows))


def make_dl_attrs(path: str, run_number: int = 10623) -> DLAttributes:
    return DLAttributes(
        path=Path(path),
        run_number=run_number,
        tailcuts_level=(10, 5),
        lstchain_version="0.11",
        nsb_level=None,
    )


def patch_finder(monkeypatch, results_by_key: dict):
    def fake_find(date, run_number, level):
        return results_by_key[(date, run_number, level)]

    monkeypatch.setattr(producers_module, "find_la_palma_files", fake_find)


RUN_10623 = {"runnumber": 10623, "date": 20220426}


class TestDL2Producer:
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

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_build_reconstructs_pending_dl1(self, tmp_path, monkeypatch, overwrite):
        (dl1_file,) = self._make_dl1_tree(tmp_path, 10623)
        output_directory = tmp_path / "out" / "dl2"
        model_directory = tmp_path / "models"
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=str(model_directory), overwrite=overwrite),
        )
        producer = DL2Producer(config, tmp_path / "out")
        calls = []

        def fake_reconstruct(files, output_directory, model_directory_template, overwrite=False):
            files = list(files)
            calls.append((files, output_directory, model_directory_template, overwrite))
            return [output_directory / "dl2_LST-1.Run10623.h5"]

        monkeypatch.setattr(producers_module, "reconstruct_dl1_to_dl2", fake_reconstruct)

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts["dl2:built"] == 1
        assert calls == [([dl1_file], output_directory, str(model_directory), overwrite)]

    def test_build_does_not_reconstruct_when_no_files_are_pending(self, tmp_path, monkeypatch):
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models"),
        )
        producer = DL2Producer(config, tmp_path / "out")
        monkeypatch.setattr(producers_module, "find_pending_dl1_files", lambda *args, **kwargs: ([], 1))

        def unexpected_reconstruct(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("reconstruction must not run without pending files")

        monkeypatch.setattr(producers_module, "reconstruct_dl1_to_dl2", unexpected_reconstruct)

        counts = producer.produce(make_selection([RUN_10623]))

        assert counts == {"dl2:existing": 1}

    def test_slurm_build_does_not_resolve_nsb_before_submission(self, tmp_path, monkeypatch):
        (dl1_file,) = self._make_dl1_tree(tmp_path, 10001)
        model_template = "/models/nsb_tuning_$NSB_LEVEL/dec_347"
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory=model_template),
        )
        producer = DL2Producer(config, tmp_path / "out", slurm_config=SlurmConfig(enabled=True))

        def unexpected_nsb_lookup(path):
            raise AssertionError(f"SLURM submission looked up NSB for {path}")

        monkeypatch.setattr("lst_tools.dl2.reconstruction.resolve_dl1_nsb_level", unexpected_nsb_lookup)
        submissions = []

        def fake_submit(self, dl1_files, model_directory_template, overwrite=False):
            submissions.append((list(dl1_files), model_directory_template, overwrite))
            return [123]

        monkeypatch.setattr(producers_module.DL2SlurmSubmitter, "submit", fake_submit)

        counts = producer.produce(make_selection([{"runnumber": 10001, "date": 20220426}]))

        assert submissions == [([dl1_file], model_template, False)]
        assert counts["dl2:slurm_jobs"] == 1
        assert counts["dl2:slurm_runs"] == 1

    def test_produce_uses_local_build_when_slurm_disabled(self, tmp_path, monkeypatch):
        config = DL2ProducerConfig(
            enabled=True,
            source="build",
            build=DL2BuildOptions(rf_directory="/models"),
        )
        producer = DL2Producer(config, tmp_path / "out", slurm_config=SlurmConfig(enabled=False))
        selection = make_selection([RUN_10623])
        expected = {"dl2:built": 1}
        monkeypatch.setattr(producer, "build", lambda selected: expected)

        assert producer.produce(selection) is expected
