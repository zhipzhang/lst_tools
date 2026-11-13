from pathlib import Path

import pandas as pd
import pytest

from lst_tools.pipeline.dl3_inputs import find_selected_dl2_files
from lst_tools.pipeline.selection import RunSelection


def make_selection(*run_numbers: int) -> RunSelection:
    return RunSelection(
        selected=pd.DataFrame(
            [{"runnumber": run_number, "date": 20220426} for run_number in run_numbers]
        )
    )


def make_dl2_file(directory: Path, run_number: int, suffix: str = "") -> Path:
    path = directory / f"dl2_LST-1.Run{run_number:05d}{suffix}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_returns_only_selected_dl2_files(tmp_path):
    dl2_directory = tmp_path / "dl2"
    expected_1 = make_dl2_file(dl2_directory, 10623)
    expected_2 = make_dl2_file(dl2_directory, 10624)
    make_dl2_file(dl2_directory, 99999)
    (dl2_directory / "dl2_unrelated.h5").touch()

    selected = find_selected_dl2_files(make_selection(10623, 10624), dl2_directory)

    assert selected == [expected_1, expected_2]


def test_warns_and_skips_missing_or_ambiguous_runs(tmp_path):
    dl2_directory = tmp_path / "dl2"
    expected = make_dl2_file(dl2_directory, 10623)
    make_dl2_file(dl2_directory, 10624)
    make_dl2_file(dl2_directory, 10624, suffix=".extra")

    with pytest.warns(UserWarning) as warnings:
        selected = find_selected_dl2_files(
            make_selection(10623, 10624, 10625),
            dl2_directory,
        )

    assert selected == [expected]
    message = str(warnings[0].message)
    assert "ambiguous DL2: 10624" in message
    assert "missing DL2: 10625" in message


def test_missing_dl2_directory_warns_and_returns_no_files(tmp_path):
    with pytest.warns(UserWarning, match="no DL2 directory") as warnings:
        selected = find_selected_dl2_files(
            make_selection(10623, 10624),
            tmp_path / "missing",
        )

    assert selected == []
    message = str(warnings[0].message)
    assert "10623" in message and "10624" in message


def test_empty_selection_does_not_require_a_dl2_directory(tmp_path):
    selected = find_selected_dl2_files(
        make_selection(),
        tmp_path / "missing",
    )

    assert selected == []
