from pathlib import Path

import pandas as pd
import pytest

from lst_tools.pipeline.dl2_inputs import find_pending_dl1_files
from lst_tools.pipeline.selection import RunSelection


def make_selection(*run_numbers: int) -> RunSelection:
    return RunSelection(
        selected=pd.DataFrame(
            [{"runnumber": run_number, "date": 20220426} for run_number in run_numbers]
        )
    )


def make_dl1_file(directory: Path, run_number: int, suffix: str = "") -> Path:
    path = directory / f"dl1_LST-1.Run{run_number:05d}{suffix}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_returns_only_selected_runs_without_existing_dl2(tmp_path):
    dl1_directory = tmp_path / "dl1"
    output_directory = tmp_path / "dl2"
    make_dl1_file(dl1_directory, 10623)
    expected = make_dl1_file(dl1_directory, 10624)
    make_dl1_file(dl1_directory, 99999)
    existing_output = output_directory / "dl2_LST-1.Run10623.h5"
    existing_output.parent.mkdir(parents=True)
    existing_output.touch()

    pending, existing_count = find_pending_dl1_files(
        make_selection(10623, 10624),
        dl1_directory,
        output_directory,
    )

    assert pending == [expected]
    assert existing_count == 1


def test_overwrite_includes_runs_with_existing_dl2(tmp_path):
    dl1_directory = tmp_path / "dl1"
    output_directory = tmp_path / "dl2"
    expected = make_dl1_file(dl1_directory, 10623)
    existing_output = output_directory / "dl2_LST-1.Run10623.h5"
    existing_output.parent.mkdir(parents=True)
    existing_output.touch()

    pending, existing_count = find_pending_dl1_files(
        make_selection(10623),
        dl1_directory,
        output_directory,
        overwrite=True,
    )

    assert pending == [expected]
    assert existing_count == 0


def test_warns_and_skips_missing_or_ambiguous_runs(tmp_path):
    dl1_directory = tmp_path / "dl1"
    expected = make_dl1_file(dl1_directory, 10623)
    make_dl1_file(dl1_directory, 10624)
    make_dl1_file(dl1_directory, 10624, suffix=".extra")

    with pytest.warns(UserWarning) as warnings:
        pending, existing_count = find_pending_dl1_files(
            make_selection(10623, 10624, 10625),
            dl1_directory,
            tmp_path / "dl2",
        )

    assert pending == [expected]
    assert existing_count == 0
    message = str(warnings[0].message)
    assert "ambiguous DL1: 10624" in message
    assert "missing DL1: 10625" in message


def test_missing_dl1_directory_warns_and_returns_no_files(tmp_path):
    with pytest.warns(UserWarning, match="no DL1 directory"):
        pending, existing_count = find_pending_dl1_files(
            make_selection(10623, 10624),
            tmp_path / "missing",
            tmp_path / "dl2",
        )

    assert pending == []
    assert existing_count == 0


def test_empty_selection_does_not_require_a_dl1_directory(tmp_path):
    pending, existing_count = find_pending_dl1_files(
        make_selection(),
        tmp_path / "missing",
        tmp_path / "dl2",
    )

    assert pending == []
    assert existing_count == 0
