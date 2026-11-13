from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import lst_tools.pipeline.producers as producers_module
from lst_tools.finder import ObservedDataMetadata
from lst_tools.pipeline import (
    DL1LinkOptions,
    DL1Producer,
    DL1ProducerConfig,
    RunSelection,
    SlurmConfig,
)


def make_selection(rows) -> RunSelection:
    return RunSelection(selected=pd.DataFrame(rows))


def make_dl_attrs(path: str, run_number: int = 10623) -> ObservedDataMetadata:
    return ObservedDataMetadata(
        path=Path(path),
        run_number=run_number,
        tailcuts_level=(10, 5),
        lstchain_version="0.11",
        nsb_level=None,
    )


def patch_finder(monkeypatch, results_by_key: dict):
    """Fake find_la_palma_run returning a per-(date, run, level) result."""

    def fake_find(date, run_number, level):
        return results_by_key[(date, run_number, level)]

    monkeypatch.setattr(producers_module, "find_la_palma_run", fake_find)


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
                (20220426, 10623, "dl1"): SimpleNamespace(newest_metadata=make_dl_attrs(source_1)),
                (20220426, 10624, "dl1"): SimpleNamespace(newest_metadata=make_dl_attrs(source_2)),
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
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_metadata=make_dl_attrs(source))})
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
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_metadata=make_dl_attrs(source))})
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
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_metadata=None)})
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
        patch_finder(monkeypatch, {(20220426, 10623, "dl1"): SimpleNamespace(newest_metadata=make_dl_attrs(source))})
        config = DL1ProducerConfig(link=DL1LinkOptions(output_directory=tmp_path / "custom_dl1"))
        producer = DL1Producer(config, tmp_path / "out")

        producer.produce(make_selection([RUN_10623]))

        assert (tmp_path / "custom_dl1" / source.name).is_symlink()
        assert not (tmp_path / "out" / "dl1").exists()

    def test_disabled_producer_does_nothing(self, tmp_path, monkeypatch):
        def forbidden_find(date, run_number, level):  # pragma: no cover - must not be called
            raise AssertionError("a disabled producer must not search the server")

        monkeypatch.setattr(producers_module, "find_la_palma_run", forbidden_find)
        producer = DL1Producer(DL1ProducerConfig(enabled=False), tmp_path / "out")

        assert producer.produce(make_selection([RUN_10623])) == {}

    def test_default_slurm_build_raises_not_implemented(self, tmp_path):
        config = DL1ProducerConfig(enabled=True, source="build")  # type: ignore[arg-type]
        producer = DL1Producer(config, tmp_path / "out")
        with pytest.raises(NotImplementedError, match="SLURM building is not supported yet"):
            producer.slurm_build(make_selection([RUN_10623]), SlurmConfig(enabled=True))
