from unittest.mock import Mock

import pytest

from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.irf.irf_nodes import IRFNode
from lst_tools.irf.utils import find_dl2_mc_path


def make_dl2_table(nsb_level=0.81, dec_line=2276, intensity_cuts=80):
    dl2_table = Mock(spec=LSTDL2EventTable)
    dl2_table.nsb_level = nsb_level
    dl2_table.dec_line = dec_line
    dl2_table.intensity_cuts = intensity_cuts
    return dl2_table


def make_mc_file(base_path, campaign_nsb, dec_line, zenith, azimuth, *, diffuse=True):
    gamma_directory = "GammaDiffuse" if diffuse else "Gamma"
    dec_name = f"dec_min_{abs(dec_line)}" if dec_line < 0 else f"dec_{dec_line}"
    node_path = (
        base_path
        / f"20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_{campaign_nsb}"
        / "TestingDataset"
        / gamma_directory
        / dec_name
        / f"node_theta_{zenith}_az_{azimuth}_"
    )
    node_path.mkdir(parents=True, exist_ok=True)
    dl2_path = node_path / f"dl2_test_node_theta_{zenith}_az_{azimuth}__merged.h5"
    dl2_path.touch()
    return dl2_path


def test_rejects_nonexistent_base_path(tmp_path):
    non_existent_path = tmp_path / "non-existent"

    with pytest.raises(FileNotFoundError):
        find_dl2_mc_path(non_existent_path, make_dl2_table())


def test_finds_all_dl2_mc_paths_for_matching_nsb_and_declination(tmp_path):
    second_path = make_mc_file(tmp_path, 0.81, 2276, 71.94, 331.291)
    first_path = make_mc_file(tmp_path, 0.81, 2276, 52.374, 133.619)

    make_mc_file(tmp_path, 0.22, 2276, 10.0, 20.0)
    make_mc_file(tmp_path, 0.81, 3476, 30.0, 40.0)
    make_mc_file(tmp_path, 0.81, 2276, 50.0, 60.0, diffuse=False)

    result = find_dl2_mc_path(tmp_path, make_dl2_table())

    assert result == [
        IRFNode(
            declination=22.76,
            zenith=52.374,
            azimuth=133.619,
            intensity_cuts=80,
            dl2_path=first_path,
        ),
        IRFNode(
            declination=22.76,
            zenith=71.94,
            azimuth=331.291,
            intensity_cuts=80,
            dl2_path=second_path,
        ),
    ]
    assert [node.dec_name for node in result] == ["dec_2276", "dec_2276"]


def test_can_find_point_like_gamma_mc(tmp_path):
    dl2_path = make_mc_file(tmp_path, 0.81, 2276, 52.374, 133.619, diffuse=False)

    result = find_dl2_mc_path(tmp_path, make_dl2_table(), diffuse=False)

    assert [node.dl2_path for node in result] == [dl2_path]


@pytest.mark.parametrize("dec_line", [2276, 3476, 4822, 6166, 6676, 931, -1802, -2924, -413])
def test_supports_declination_directory_names(tmp_path, dec_line):
    dl2_path = make_mc_file(tmp_path, 0.81, dec_line, 52.374, 133.619)

    result = find_dl2_mc_path(tmp_path, make_dl2_table(dec_line=dec_line))

    assert len(result) == 1
    assert result[0].dl2_path == dl2_path
    assert result[0].declination == dec_line / 100
    assert result[0].dec_name == dl2_path.parents[1].name
