import dataclasses

import pandas as pd
import pytest

from lst_tools.pipeline import RunSelection, SelectedRun


def make_selection(rows: list[dict], edges=(0.0, 20.0, 30.0)) -> RunSelection:
    frame = pd.DataFrame(rows)
    return RunSelection(edges=edges, selected=frame)


BINNED = [
    {"run_number": 10623, "date": 20220426, "zenith_bin": "zd_20_30"},
    {"run_number": 10624, "date": 20220426, "zenith_bin": "zd_0_20"},
]
UNBINNED = {"run_number": 10625, "date": 20220427, "zenith_bin": None}


def test_iteration_yields_typed_selected_runs():
    selection = make_selection(BINNED)

    runs = list(selection)

    assert runs == [
        SelectedRun(run_number=10623, date=20220426, zenith_bin="zd_20_30"),
        SelectedRun(run_number=10624, date=20220426, zenith_bin="zd_0_20"),
    ]
    assert all(isinstance(run.run_number, int) for run in runs)
    assert all(isinstance(run.date, int) for run in runs)


def test_unbinned_runs_are_excluded_from_iteration_and_listed():
    selection = make_selection([*BINNED, UNBINNED])

    assert [run.run_number for run in selection] == [10623, 10624]
    assert selection.unbinned_run_numbers == [10625]


def test_all_unbinned_selection_iterates_empty():
    selection = make_selection([UNBINNED])

    assert list(selection) == []
    assert selection.unbinned_run_numbers == [10625]


def test_selected_run_is_immutable():
    run = SelectedRun(run_number=1, date=20220101, zenith_bin="zd_0_20")

    with pytest.raises(dataclasses.FrozenInstanceError):
        run.run_number = 2


def test_repr_summarizes_counts_and_edges():
    selection = make_selection([*BINNED, UNBINNED])

    assert repr(selection) == "RunSelection(2 binned runs, 1 unbinned, edges=(0.0, 20.0, 30.0))"
