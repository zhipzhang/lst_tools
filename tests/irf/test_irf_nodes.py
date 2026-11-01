from pathlib import Path

import pytest

from lst_tools.event_filter import EventFilter
from lst_tools.irf import IRFNode, MCNode, irf_group_path


def make_mc_node(irf_type="full-enclosure"):
    return MCNode(
        nsb_level=0.81,
        dec_line=2276,
        zenith=52.374,
        azimuth=133.619,
        dl2_path=Path("dl2_merged.h5"),
        irf_type=irf_type,
    )


def make_node(irf_type="full-enclosure", gh_type="efficiency", gh_value=0.7):
    return IRFNode(
        nsb_level=0.81,
        dec_line=2276,
        zenith=52.374,
        azimuth=133.619,
        dl2_path=Path("dl2_merged.h5"),
        irf_type=irf_type,
        gh_type=gh_type,
        gh_value=gh_value,
    )


def test_mc_node_names_describe_the_input_pointing():
    node = make_mc_node()

    assert node.declination == pytest.approx(22.76)
    assert node.dec_name == "dec_2276"
    assert node.pointing_name == "theta_52.374_az_133.619"


def test_mc_node_has_no_irf_output_naming():
    assert not hasattr(make_mc_node(), "relative_path")


def test_irf_node_names_describe_the_output_file():
    node = make_node()

    assert node.file_name == "theta_52.374_az_133.619.irf.fits.gz"
    assert node.relative_path == Path("nsb_0.81/dec_2276/full-enclosure/gheff0.7/theta_52.374_az_133.619.irf.fits.gz")


def test_irf_node_makes_point_type_explicit_in_output_path():
    node = make_node("point")

    assert node.relative_path == Path("nsb_0.81/dec_2276/point/gheff0.7/theta_52.374_az_133.619.irf.fits.gz")


@pytest.mark.parametrize(
    ("gh_type", "gh_value", "label"),
    [
        ("efficiency", 0.7, "gheff0.7"),
        ("efficiency", 0.9, "gheff0.9"),
        ("cut", 0.7, "ghcut0.7"),
        ("cut", 0.5, "ghcut0.5"),
    ],
)
def test_cut_label_distinguishes_gammaness_selections(gh_type, gh_value, label):
    node = make_node(gh_type=gh_type, gh_value=gh_value)

    assert node.cut_label == label
    assert node.relative_path.parent.name == label


def test_cut_label_separates_otherwise_identical_irfs():
    efficiency = make_node(gh_type="efficiency", gh_value=0.7)
    fixed = make_node(gh_type="cut", gh_value=0.7)

    assert efficiency.relative_path != fixed.relative_path


def test_from_mc_node_copies_the_simulation_and_adds_the_cuts():
    mc_node = make_mc_node("point")

    node = IRFNode.from_mc_node(mc_node, gh_type="cut", gh_value=0.6)

    assert isinstance(node, IRFNode)
    assert node.dl2_path == mc_node.dl2_path
    assert node.zenith == mc_node.zenith
    assert node.irf_type == "point"
    assert node.gh_type == "cut"
    assert node.gh_value == pytest.approx(0.6)


def test_from_mc_node_defaults_to_the_gammaness_efficiency():
    node = IRFNode.from_mc_node(make_mc_node())

    assert node.gh_type == "efficiency"
    assert node.gh_value == pytest.approx(0.7)


def test_irf_group_path_is_the_directory_above_the_pointing_files():
    group = irf_group_path(nsb_level=0.38, dec_line=-2276, irf_type="point", cut_label="gheff0.7")

    assert group == Path("nsb_0.38/dec_min_2276/point/gheff0.7")


def test_node_relative_path_lives_in_the_group_directory():
    node = make_node()

    assert node.relative_path.parent == irf_group_path(
        nsb_level=node.nsb_level,
        dec_line=node.dec_line,
        irf_type=node.irf_type,
        cut_label=node.cut_label,
    )


@pytest.mark.parametrize(
    ("gh_type", "gh_value"),
    [("efficiency", 0.7), ("cut", 0.7), ("efficiency", 0.9)],
)
def test_node_and_event_filter_agree_on_the_cut_label(gh_type, gh_value):
    if gh_type == "efficiency":
        event_filter = EventFilter.from_cuts(80, 1.0, gh_efficiency=gh_value)
    else:
        event_filter = EventFilter.from_cuts(80, 1.0, gh_cut=gh_value)

    assert make_node(gh_type=gh_type, gh_value=gh_value).cut_label == event_filter.cut_label
