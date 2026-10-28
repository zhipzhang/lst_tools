from pathlib import Path

import pytest

from lst_tools.irf import IRFNode


def make_node(irf_type="full-enclosure"):
    return IRFNode(
        nsb_level=0.81,
        dec_line=2276,
        zenith=52.374,
        azimuth=133.619,
        dl2_path=Path("dl2_merged.h5"),
        irf_type=irf_type,
    )


def test_irf_node_names_describe_the_output_file():
    node = make_node()

    assert node.declination == pytest.approx(22.76)
    assert node.pointing_name == "theta_52.374_az_133.619"
    assert node.file_name == "theta_52.374_az_133.619.irf.fits.gz"
    assert node.relative_path == Path(
        "nsb_0.81/dec_2276/full-enclosure/theta_52.374_az_133.619.irf.fits.gz"
    )


def test_irf_node_makes_point_type_explicit_in_output_path():
    node = make_node("point")

    assert node.relative_path == Path("nsb_0.81/dec_2276/point/theta_52.374_az_133.619.irf.fits.gz")
