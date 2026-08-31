from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.dl2 import dl2table as dl2table_module


@pytest.fixture
def patched_dl2_io(monkeypatch):
    events = pd.DataFrame(
        {
            "intensity": [100.0, 150.0, 200.0],
            "alt_tel": [1.0, 1.1, 1.2],
            # The middle-event convention remains correct across the 2π wrap.
            "az_tel": [2 * np.pi - 0.01, 0.0001, 0.01],
        }
    )
    provenance = {
        "input": [
            {
                "role": ["input"],
                "url": "/data/tailcut1005/dl1_LST-1.Run00042.h5",
            },
            {
                "role": ["model"],
                "url": "/models/nsb_tuning_0.22/dec_3476",
            },
        ]
    }
    calls = {}

    def read_hdf(filename, key):
        calls["hdf"] = filename, key
        return events

    def get_effective_time(data):
        assert data is events
        return 12.5, 15.0

    monkeypatch.setattr(dl2table_module.pd, "read_hdf", read_hdf)
    monkeypatch.setattr(dl2table_module, "get_effective_time", get_effective_time)
    monkeypatch.setattr(dl2table_module, "read_dl2_provenance", lambda filename: provenance)
    monkeypatch.setattr(dl2table_module, "get_intensity_cut", lambda data: 21.0)

    return events, calls


def test_loads_events_pointing_and_provenance(patched_dl2_io):
    events, calls = patched_dl2_io
    filename = Path("/data/dl2_LST-1.Run00042.h5")

    table = LSTDL2EventTable(filename)

    assert table.run_id == 42
    assert table.pointing_alt == pytest.approx(1.1)
    assert table.pointing_az == pytest.approx(0.0001)
    assert table.pointing_zen == pytest.approx(np.pi / 2 - 1.1)
    assert table.data is events
    assert table.t_eff == pytest.approx(12.5)
    assert table.t_elapsed == pytest.approx(15.0)
    assert calls["hdf"][0] == str(filename)


def test_parses_processing_metadata(patched_dl2_io):
    table = LSTDL2EventTable("/data/dl2_LST-1.Run00042.h5")

    assert table.tailcut_level == (10, 5)
    assert table.nsb_level == pytest.approx(0.22)
    assert table.dec_line == 3476
    assert table.intensity_cuts == pytest.approx(30.0)


def test_rejects_filename_without_run_number_before_reading_file(monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("DL2 file should not be read when its run number is invalid")

    monkeypatch.setattr(dl2table_module.pd, "read_hdf", unexpected_read)

    with pytest.raises(ValueError, match="run number"):
        LSTDL2EventTable("/data/dl2_without_run_number.h5")


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (pd.DataFrame(), "contains no events"),
        (pd.DataFrame({"alt_tel": [1.0]}), "missing required columns"),
    ],
)
def test_rejects_dl2_data_without_pointing(events, message, monkeypatch):
    monkeypatch.setattr(dl2table_module.pd, "read_hdf", lambda *args, **kwargs: events)

    with pytest.raises(ValueError, match=message):
        LSTDL2EventTable("/data/dl2_LST-1.Run00042.h5")
