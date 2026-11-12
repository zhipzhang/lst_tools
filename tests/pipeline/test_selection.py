import dataclasses

import pandas as pd
import pytest

from lst_tools.pipeline import RunSelection, SelectedRun


def make_selection(rows: list[dict]) -> RunSelection:
    return RunSelection(selected=pd.DataFrame(rows))


RUNS = [
    {"run_number": 10623, "date": 20220426},
    {"run_number": 10624, "date": 20220426},
]


def test_iteration_yields_every_selected_run():
    runs = list(make_selection(RUNS))

    assert runs == [
        SelectedRun(run_number=10623, date=20220426),
        SelectedRun(run_number=10624, date=20220426),
    ]
    assert all(isinstance(run.run_number, int) for run in runs)
    assert all(isinstance(run.date, int) for run in runs)


def test_selected_run_is_immutable():
    run = SelectedRun(run_number=1, date=20220101)

    with pytest.raises(dataclasses.FrozenInstanceError):
        run.run_number = 2


def test_repr_summarizes_run_count():
    assert repr(make_selection(RUNS)) == "RunSelection(2 runs)"


def test_empty_selection_iterates_and_reprs():
    selection = RunSelection(selected=pd.DataFrame({"run_number": [], "date": []}))

    assert list(selection) == []
    assert repr(selection) == "RunSelection(0 runs)"
