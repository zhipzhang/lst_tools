import pytest

from lst_tools.dl1 import nsb as nsb_module
from lst_tools.dl1.file import LSTDL1File


def test_construction_parses_path_metadata_without_reading_dl1_data(make_dl1_path, monkeypatch):
    def unexpected_resolution(*args):
        raise AssertionError("constructing LSTDL1File must not read DL1 data")

    monkeypatch.setattr(nsb_module, "find_tailcuts", unexpected_resolution)
    path = make_dl1_path(10623)

    dl1 = LSTDL1File(path)

    assert dl1.file_name == path.resolve()
    assert dl1.run_number == 10623
    assert dl1.tailcuts_level == (10, 5)
    assert dl1.lstchain_version == "0.10"


def test_nsb_level_is_resolved_once_and_reused(make_dl1_path, monkeypatch):
    path = make_dl1_path()
    calls = []

    def fake_find_tailcuts(directory, run_number):
        calls.append(run_number)
        return 12.5, 0.25, None

    monkeypatch.setattr(nsb_module, "find_tailcuts", fake_find_tailcuts)
    dl1 = LSTDL1File(path)

    assert dl1.nsb_level == pytest.approx(0.38)
    assert dl1.nsb_level == pytest.approx(0.38)
    assert calls == [10623]


def test_nsb_level_cache_is_per_file_instance(make_dl1_path, monkeypatch):
    first_path = make_dl1_path(10623)
    second_path = make_dl1_path(10624)
    nsb_by_run = {10623: 0.25, 10624: 0.70}
    calls = []

    def fake_find_tailcuts(directory, run_number):
        calls.append(run_number)
        return 12.5, nsb_by_run[run_number], None

    monkeypatch.setattr(nsb_module, "find_tailcuts", fake_find_tailcuts)

    assert LSTDL1File(first_path).nsb_level == pytest.approx(0.38)
    assert LSTDL1File(second_path).nsb_level == pytest.approx(0.81)
    assert calls == [10623, 10624]


def test_failed_nsb_resolution_is_not_cached_and_is_retried(make_dl1_path, monkeypatch):
    path = make_dl1_path()
    outcomes = iter([ValueError("Could not parse nsb level"), (12.5, 0.25, None)])

    def flaky_find_tailcuts(directory, run_number):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(nsb_module, "find_tailcuts", flaky_find_tailcuts)
    dl1 = LSTDL1File(path)

    with pytest.raises(RuntimeError, match="Failed to resolve NSB level: Could not parse nsb level"):
        _ = dl1.nsb_level

    assert dl1.nsb_level == pytest.approx(0.38)


def test_invalid_path_metadata_fails_during_construction(tmp_path):
    path = tmp_path / "dl1_LST-1.Run10623.h5"
    path.touch()

    with pytest.raises(ValueError, match="Could not parse tailcuts level"):
        LSTDL1File(path)


def test_missing_file_fails_during_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        LSTDL1File(tmp_path / "missing.h5")
