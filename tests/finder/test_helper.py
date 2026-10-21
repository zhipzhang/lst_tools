import pytest

from lst_tools.finder import NSBLevel
from lst_tools.finder.helper import (
    parse_dl3_path,
    parse_gh_eff,
    parse_lstchain_version,
    parse_nsb_level_from_path,
    parse_run_id,
    parse_tailcuts_level,
    parse_theta_containment,
)


def test_parse_run_id():
    assert parse_run_id("muons_LST-1.Run10615.0711.fits") == 10615
    assert parse_run_id("dl1_LST-1.Run10623.h5") == 10623
    with pytest.raises(ValueError):
        parse_run_id("")


def test_parse_tailcuts_level():
    assert parse_tailcuts_level("tailcut12") == (1, 2)
    assert parse_tailcuts_level("tailcut1206") == (12, 6)
    assert parse_tailcuts_level("tailcut120607") == (120, 607)
    with pytest.raises(ValueError):
        parse_tailcuts_level("")


def test_parse_nsb_level_from_path():
    assert parse_nsb_level_from_path("nsb_tuning_0.38") == NSBLevel(0.38)
    with pytest.raises(ValueError):
        parse_nsb_level_from_path("nsb_tuning_0.999")


def test_parse_gh_eff():
    assert parse_gh_eff("anythings/gheff0.7_xxxxx") == 0.7


def test_parse_theta_containment():
    assert parse_theta_containment("anythings/gheff0.7_xxx_thetacont0.9/yyyy") == 0.9


def test_parse_lstchain_version():
    assert parse_lstchain_version("v10.10") == "10.10"
    assert parse_lstchain_version("/anythings/v0.9") == "0.9"


DL3_EXAMPLE_PATH = (
    "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821/v0.10/tailcut1005/nsb_tuning_0.38/"
    "0.10.18/20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.38/src_indep/"
    "full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"
)


def test_parse_dl3_path():
    attrs = parse_dl3_path(DL3_EXAMPLE_PATH)

    assert attrs.run_number == 14083
    assert attrs.tailcuts_level == (10, 5)
    assert attrs.lstchain_version == "0.10"
    assert attrs.nsb_level == NSBLevel(0.38)
    assert attrs.irf_type == "full-enclosure"
    assert attrs.mc_data_type == "ring-wobble"
    assert attrs.gh_eff == 0.7
    assert attrs.theta_containment == 0.7


def test_parse_dl3_path_point_diffuse():
    path = DL3_EXAMPLE_PATH.replace("full-enclosure", "point").replace("ring-wobble", "diffuse")
    attrs = parse_dl3_path(path)

    assert attrs.irf_type == "point"
    assert attrs.mc_data_type == "diffuse"


def test_parse_dl3_path_missing_irf_type():
    path = DL3_EXAMPLE_PATH.replace("full-enclosure/", "")
    with pytest.raises(ValueError):
        parse_dl3_path(path)


def test_parse_dl3_path_component_matching_not_substring():
    # "pointing" contains "point" but is not an irf type component
    path = DL3_EXAMPLE_PATH.replace("full-enclosure", "pointing")
    with pytest.raises(ValueError):
        parse_dl3_path(path)
