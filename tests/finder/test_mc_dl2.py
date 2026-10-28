from pathlib import Path

import pytest

from lst_tools.finder import MCDL2Attribute, find_mc_dl2_files, parse_mc_dl2_path

EXAMPLE_PATH = Path(
    "/fefs/aswg/data/mc/DL2/AllSky/20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.14/"
    "TestingDataset/Gamma/dec_2276/node_theta_10.0_az_248.117_/"
    "dl2_20240918_v0.10.12_allsky_nsb_tuning_0.14_Gamma_test_node_theta_10.0_az_248.117__merged.h5"
)


def make_mc_file(campaign, dec, theta, azimuth, *, mc_type="GammaDiffuse", corsika_node=False):
    dec_name = f"dec_min_{abs(dec)}" if dec < 0 else f"dec_{dec}"
    node_prefix = "node_corsika_theta" if corsika_node else "node_theta"
    node_path = campaign / "TestingDataset" / mc_type / dec_name / f"{node_prefix}_{theta}_az_{azimuth}_"
    node_path.mkdir(parents=True, exist_ok=True)
    path = node_path / f"dl2_test_node_theta_{theta}_az_{azimuth}__merged.h5"
    path.touch()
    return path


def test_parse_mc_dl2_path_reads_attributes_from_directories():
    attributes = parse_mc_dl2_path(EXAMPLE_PATH)

    assert attributes == MCDL2Attribute(
        path=EXAMPLE_PATH,
        mc_type="Gamma",
        theta=10.0,
        azimuth=248.117,
        nsb=0.14,
        dec=2276,
    )


def test_parse_mc_dl2_path_handles_negative_dec_and_corsika_nodes(tmp_path):
    campaign = tmp_path / "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.22"
    path = make_mc_file(campaign, -1802, 12.829, 301.263, corsika_node=True)

    attributes = parse_mc_dl2_path(path)

    assert attributes.dec == -1802
    assert attributes.theta == pytest.approx(12.829)
    assert attributes.azimuth == pytest.approx(301.263)
    assert attributes.nsb == pytest.approx(0.22)


def test_parse_mc_dl2_path_rejects_nonstandard_layout(tmp_path):
    with pytest.raises(ValueError, match="TestingDataset"):
        parse_mc_dl2_path(tmp_path / "dl2_test_merged.h5")


def test_find_mc_dl2_files_recursively_returns_sorted_attributes(tmp_path):
    campaign = tmp_path / "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.81"
    diffuse_path = make_mc_file(campaign, 2276, 52.374, 133.619)
    gamma_path = make_mc_file(campaign, 2276, 10.0, 248.117, mc_type="Gamma")
    (campaign / "unrelated.h5").touch()

    attributes = find_mc_dl2_files(campaign)

    assert [attribute.path for attribute in attributes] == sorted([diffuse_path, gamma_path], key=str)
    assert {attribute.mc_type for attribute in attributes} == {"Gamma", "GammaDiffuse"}


def test_find_mc_dl2_files_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_mc_dl2_files(tmp_path / "missing")


def test_find_mc_dl2_files_rejects_file_base(tmp_path):
    path = tmp_path / "campaign"
    path.touch()

    with pytest.raises(NotADirectoryError):
        find_mc_dl2_files(path)
