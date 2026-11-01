from pathlib import Path
from types import SimpleNamespace

import pytest

from lst_tools.dl3 import DL3Reducer
from lst_tools.event_filter import EventFilter
from lst_tools.irf import IRFConfig, IRFNode


@pytest.mark.parametrize(
    ("point_like", "irf_type"),
    [(False, "full-enclosure"), (True, "point")],
)
def test_irf_set_dir_includes_irf_type_and_cut_label(tmp_path, monkeypatch, point_like, irf_type):
    monkeypatch.setattr(
        "lst_tools.dl3.reduction.LSTDL2EventTable",
        lambda path: SimpleNamespace(nsb_level=0.81, dec_line=2276),
    )
    irf_config = IRFConfig(EventFilter.from_cuts(80, 1.0, gh_cut=0.7), point_like=point_like)
    reducer = DL3Reducer(tmp_path, "Crab", 83.6, 22.0, irf_config)
    irf_directory = tmp_path / "nsb_0.81" / "dec_2276" / irf_type / "ghcut0.7"
    irf_directory.mkdir(parents=True)

    assert reducer._irf_set_dir(Path("dl2_LST-1.Run00001.h5")) == irf_directory


def test_irf_set_dir_is_the_directory_the_generator_writes(tmp_path, monkeypatch):
    # The reduction and the generation must agree on the layout without
    # restating it: the reducer's lookup is the node's own directory.
    monkeypatch.setattr(
        "lst_tools.dl3.reduction.LSTDL2EventTable",
        lambda path: SimpleNamespace(nsb_level=0.38, dec_line=-2276),
    )
    irf_config = IRFConfig(EventFilter.from_cuts(80, 1.0, gh_efficiency=0.7), point_like=True)
    reducer = DL3Reducer(tmp_path, "Crab", 83.6, 22.0, irf_config)
    node = IRFNode(
        nsb_level=0.38,
        dec_line=-2276,
        zenith=20,
        azimuth=180,
        dl2_path=Path("dl2_mc_merged.h5"),
        irf_type="point",
        gh_type="efficiency",
        gh_value=0.7,
    )
    (tmp_path / node.relative_path.parent).mkdir(parents=True)

    assert reducer._irf_set_dir(Path("dl2_LST-1.Run00001.h5")) == tmp_path / node.relative_path.parent
