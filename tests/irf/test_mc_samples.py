from pathlib import Path

import pytest

from lst_tools.finder import MCDL2Metadata
from lst_tools.irf import GammaMCSample, irf_file_path, irf_grid_path, select_gamma_mc_samples


def make_sample(mc_type="diffuse"):
    return GammaMCSample(
        path=Path("dl2_merged.h5"),
        mc_type=mc_type,
        nsb_level=0.81,
        dec_line=2276,
        zenith=52.374,
        azimuth=133.619,
    )


def test_mc_sample_describes_simulation_input_and_pointing():
    sample = make_sample()

    assert sample.declination == pytest.approx(22.76)
    assert sample.pointing_name == "theta_52.374_az_133.619"
    assert sample.mc_type == "diffuse"


def test_irf_file_path_combines_sample_pointing_with_configuration():
    sample = make_sample()

    path = irf_file_path(sample, irf_type="full-enclosure", cut_label="gheff0.7")

    assert path == Path("dec_2276/nsb_0.81/full-enclosure/gheff0.7/theta_52.374_az_133.619.irf.fits.gz")


def test_irf_file_path_separates_configurations_for_the_same_sample():
    sample = make_sample()

    fixed = irf_file_path(sample, irf_type="point", cut_label="ghcut0.7_thetacut0.2")
    efficiency = irf_file_path(sample, irf_type="point", cut_label="gheff0.7_thetacont0.7")

    assert fixed != efficiency
    assert fixed.parent.name == "ghcut0.7_thetacut0.2"
    assert efficiency.parent.name == "gheff0.7_thetacont0.7"


def test_irf_grid_path_is_the_directory_above_the_pointing_files():
    cut_label = "gheff0.7_thetacont0.7"
    grid = irf_grid_path(nsb_level=0.38, dec_line=-2276, irf_type="point", cut_label=cut_label)

    assert grid == Path(f"dec_min_2276/nsb_0.38/point/{cut_label}")
    sample = make_sample()
    assert irf_file_path(sample, irf_type="point", cut_label=cut_label).parent == irf_grid_path(
        nsb_level=sample.nsb_level,
        dec_line=sample.dec_line,
        irf_type="point",
        cut_label=cut_label,
    )


def test_select_gamma_mc_samples_filters_and_deduplicates_by_tuning_and_irf_type():
    diffuse = MCDL2Metadata(Path("/mc/diffuse.h5"), "GammaDiffuse", 20, 180, 0.38, 347)
    point = MCDL2Metadata(Path("/mc/gamma.h5"), "Gamma", 20, 180, 0.38, 347)
    other_dec = MCDL2Metadata(Path("/mc/other_dec.h5"), "GammaDiffuse", 20, 180, 0.38, 999)
    other_nsb = MCDL2Metadata(Path("/mc/other_nsb.h5"), "GammaDiffuse", 20, 180, 0.81, 347)
    attributes = [diffuse, diffuse, point, other_dec, other_nsb]

    full_samples = select_gamma_mc_samples(
        attributes,
        nsb_level=0.38,
        dec_line=347,
        irf_type="full-enclosure",
    )
    point_samples = select_gamma_mc_samples(
        attributes,
        nsb_level=0.38,
        dec_line=347,
        irf_type="point",
    )

    assert full_samples == (GammaMCSample(Path("/mc/diffuse.h5"), "diffuse", 0.38, 347, 20, 180),)
    assert point_samples == (GammaMCSample(Path("/mc/gamma.h5"), "point-source", 0.38, 347, 20, 180),)
