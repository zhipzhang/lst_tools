from pathlib import Path
from types import SimpleNamespace

import pytest

from lst_tools.dl3 import DL3Reducer
from lst_tools.irf import GammaMCSample, IRFConfig, irf_file_path


def make_config(*, point_like=False, energy_dependent_gh=False):
    gh = {"gh_efficiency": 0.7} if energy_dependent_gh else {"gh_cut": 0.7}
    theta = {"theta_containment": 0.7} if energy_dependent_gh else {"theta_cut": 0.2}
    return IRFConfig.from_cuts(
        80,
        1.0,
        **gh,
        point_like=point_like,
        **(theta if point_like else {}),
    )


@pytest.mark.parametrize(
    ("point_like", "irf_type", "cut_label"),
    [
        (False, "full-enclosure", "ghcut0.7"),
        (True, "point", "ghcut0.7_thetacut0.2"),
    ],
)
def test_irf_grid_dir_includes_irf_type_and_cut_label(tmp_path, monkeypatch, point_like, irf_type, cut_label):
    monkeypatch.setattr(
        "lst_tools.dl3.reduction.LSTObservedDL2Table",
        lambda path: SimpleNamespace(nsb_level=0.81, dec_line=2276),
    )
    irf_config = make_config(point_like=point_like)
    reducer = DL3Reducer(tmp_path, tmp_path / "dl3", "Crab", 83.6, 22.0, irf_config)
    irf_directory = tmp_path / "dec_2276" / "nsb_0.81" / irf_type / cut_label
    irf_directory.mkdir(parents=True)

    assert reducer._irf_grid_dir(Path("dl2_LST-1.Run00001.h5")) == irf_directory


def test_irf_grid_dir_is_the_directory_the_generator_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lst_tools.dl3.reduction.LSTObservedDL2Table",
        lambda path: SimpleNamespace(nsb_level=0.38, dec_line=-2276),
    )
    irf_config = make_config(point_like=True, energy_dependent_gh=True)
    reducer = DL3Reducer(tmp_path, tmp_path / "dl3", "Crab", 83.6, 22.0, irf_config)
    sample = GammaMCSample(
        path=Path("dl2_mc_merged.h5"),
        mc_type="point-source",
        nsb_level=0.38,
        dec_line=-2276,
        zenith=20,
        azimuth=180,
    )
    relative_path = irf_file_path(sample, irf_type=irf_config.irf_type, cut_label=irf_config.cut_label)
    (tmp_path / relative_path.parent).mkdir(parents=True)

    assert reducer._irf_grid_dir(Path("dl2_LST-1.Run00001.h5")) == tmp_path / relative_path.parent


def test_reducer_writes_to_configured_output_dir(tmp_path):
    output_dir = tmp_path / "dl3"
    output_dir.mkdir()
    existing_dl3 = output_dir / "dl3_LST-1.Run00001.fits"
    existing_dl3.touch()
    reducer = DL3Reducer(
        tmp_path / "irfs",
        output_dir,
        "Crab",
        83.6,
        22.0,
        make_config(),
    )

    assert reducer.reduce([Path("dl2_LST-1.Run00001.h5")]) == [existing_dl3]
