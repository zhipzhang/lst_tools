import json
from pathlib import Path

import pytest

import lst_tools.irf
from lst_tools.irf import IRFConfig


def test_default_irf_config_reproduces_default_config_json():
    with open(Path(lst_tools.irf.__file__).parent / "default_config.json") as file_handle:
        reference = json.load(file_handle)

    config = IRFConfig.from_cuts(gh_cut=0.7).to_irf_config()

    assert config == {
        **reference,
        "IRFFITSWriter": {
            "energy_dependent_gh": False,
            "point_like": False,
        },
    }


def test_from_cuts_requires_exactly_one_gammaness_mode():
    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        IRFConfig.from_cuts()
    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        IRFConfig.from_cuts(gh_cut=0.7, gh_efficiency=0.7)


@pytest.mark.parametrize("gh_efficiency", [0.0, -0.5, 1.5])
def test_from_cuts_validates_gh_efficiency(gh_efficiency):
    with pytest.raises(ValueError, match="gh_efficiency must be in"):
        IRFConfig.from_cuts(gh_efficiency=gh_efficiency)


def test_full_enclosure_rejects_theta_cuts():
    with pytest.raises(ValueError, match="only valid when point_like=True"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=False, theta_cut=0.2)
    with pytest.raises(ValueError, match="only valid when point_like=True"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=False, theta_containment=0.7)


def test_point_like_requires_exactly_one_theta_mode():
    with pytest.raises(ValueError, match="exactly one of theta_cut and theta_containment"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=True)
    with pytest.raises(ValueError, match="exactly one of theta_cut and theta_containment"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=True, theta_cut=0.2, theta_containment=0.7)


@pytest.mark.parametrize("theta_containment", [0.0, -0.2, 1.2])
def test_point_like_validates_theta_containment(theta_containment):
    with pytest.raises(ValueError, match="theta_containment must be in"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=True, theta_containment=theta_containment)


@pytest.mark.parametrize("theta_cut", [0.0, -0.1])
def test_point_like_validates_theta_cut(theta_cut):
    with pytest.raises(ValueError, match="theta_cut must be positive"):
        IRFConfig.from_cuts(gh_efficiency=0.7, point_like=True, theta_cut=theta_cut)


@pytest.mark.parametrize(
    ("gammaness", "expected"),
    [
        ({"gh_cut": 0.7}, "ghcut0.7"),
        ({"gh_efficiency": 0.7}, "gheff0.7"),
        ({"gh_efficiency": 0.9}, "gheff0.9"),
    ],
)
def test_cut_label_comes_from_irf_config(gammaness, expected):
    config = IRFConfig.from_cuts(**gammaness)

    assert config.cut_label == expected


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        ({"theta_cut": 0.2}, "ghcut0.7_thetacut0.2"),
        ({"theta_cut": 0.15}, "ghcut0.7_thetacut0.15"),
        ({"theta_containment": 0.68}, "ghcut0.7_thetacont0.68"),
    ],
)
def test_point_like_cut_label_includes_theta_selection(theta, expected):
    config = IRFConfig.from_cuts(gh_cut=0.7, point_like=True, **theta)

    assert config.cut_label == expected


def test_point_like_cut_label_combines_energy_dependent_selections():
    config = IRFConfig.from_cuts(
        gh_efficiency=0.9,
        point_like=True,
        theta_containment=0.68,
    )

    assert config.cut_label == "gheff0.9_thetacont0.68"


def test_to_irf_config_and_to_dl3_config_export_expected_sections():
    config = IRFConfig.from_cuts(
        intensity_cut=80,
        leakage_cut=0.5,
        gh_efficiency=0.7,
        point_like=True,
        theta_containment=0.68,
    )

    irf_dict = config.to_irf_config()
    dl3_dict = config.to_dl3_config()

    assert set(irf_dict) == {"EventSelector", "DL3Cuts", "IRFFITSWriter", "DataBinning"}
    assert set(dl3_dict) == {"EventSelector", "DL3Cuts"}
    assert irf_dict["EventSelector"]["filters"]["intensity"] == [80.0, float("inf")]
    assert irf_dict["EventSelector"]["filters"]["leakage_intensity_width_2"] == [0, 0.5]
    assert irf_dict["DL3Cuts"]["gh_efficiency"] == pytest.approx(0.7)
    assert irf_dict["DL3Cuts"]["theta_containment"] == pytest.approx(0.68)
    assert irf_dict["IRFFITSWriter"] == {
        "energy_dependent_gh": True,
        "point_like": True,
        "energy_dependent_theta": True,
    }


def test_write_irf_config_and_from_json_round_trip(tmp_path):
    config = IRFConfig.from_cuts(
        intensity_cut=100,
        leakage_cut=0.3,
        gh_cut=0.8,
        point_like=True,
        theta_cut=0.15,
    )
    json_path = tmp_path / "nested" / "irf.config.json"

    written = config.write_irf_config(json_path)
    reloaded = IRFConfig.from_json(written)

    assert written == json_path
    assert reloaded.to_irf_config() == config.to_irf_config()
    assert reloaded.to_dl3_config() == config.to_dl3_config()
