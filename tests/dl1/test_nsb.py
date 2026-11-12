import pytest

from lst_tools.dl1 import nsb as nsb_module
from lst_tools.dl1.nsb import resolve_dl1_nsb_level
from lst_tools.finder import NSBLevel


def test_resolve_dl1_nsb_level_reads_run_directory(make_dl1_path, monkeypatch):
    path = make_dl1_path(10623)
    calls = []

    def fake_find_tailcuts(directory, run_number):
        calls.append((directory, run_number))
        return 12.5, 0.25, None

    monkeypatch.setattr(nsb_module, "find_tailcuts", fake_find_tailcuts)

    nsb_level = resolve_dl1_nsb_level(path)

    assert nsb_level == NSBLevel.L0_38
    assert calls == [(path.parent, 10623)]


def test_resolve_dl1_nsb_level_rejects_level_above_supported_range(make_dl1_path, monkeypatch):
    path = make_dl1_path()
    monkeypatch.setattr(nsb_module, "find_tailcuts", lambda directory, run_number: (12.5, 9.0, None))

    with pytest.raises(ValueError, match="No NSBLevel found higher than 9.0"):
        resolve_dl1_nsb_level(path)
