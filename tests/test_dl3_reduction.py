from pathlib import Path
from types import SimpleNamespace

import pytest

from lst_tools.dl3 import DL3Reducer
from lst_tools.event_filter import EventFilter
from lst_tools.irf import IRFConfig


@pytest.mark.parametrize(
    ("point_like", "irf_type"),
    [(False, "full-enclosure"), (True, "point")],
)
def test_irf_set_dir_includes_irf_type(tmp_path, monkeypatch, point_like, irf_type):
    monkeypatch.setattr(
        "lst_tools.dl3.reduction.LSTDL2EventTable",
        lambda path: SimpleNamespace(nsb_level=0.81, dec_line=2276),
    )
    irf_config = IRFConfig(EventFilter.from_cuts(80, 1.0, gh_cut=0.7), point_like=point_like)
    reducer = DL3Reducer(tmp_path, "Crab", 83.6, 22.0, irf_config)
    irf_directory = tmp_path / "nsb_0.81" / "dec_2276" / irf_type
    irf_directory.mkdir(parents=True)

    assert reducer._irf_set_dir(Path("dl2_LST-1.Run00001.h5")) == irf_directory
