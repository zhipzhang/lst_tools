import pytest

from lst_tools.dl2 import LSTDL2MCTable

MC_FILE_NAME = (
    "/fefs/aswg/data/mc/DL2/AllSky/20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.22"
    "/TestingDataset/Gamma/dec_2276/node_theta_10.0_az_102.199_"
    "/dl2_20240918_v0.10.12_allsky_nsb_tuning_0.22_Gamma_test_node_theta_10.0_az_102.199__merged.h5"
)


def test_parses_pointing_node():
    table = object.__new__(LSTDL2MCTable)
    table.file_name = MC_FILE_NAME

    assert table.pointing_ze.to_value("deg") == pytest.approx(10.0)
    assert table.pointing_az.to_value("deg") == pytest.approx(102.199)


def test_rejects_path_without_pointing_node():
    table = object.__new__(LSTDL2MCTable)
    table.file_name = "/data/dl2_Gamma_test_merged.h5"

    with pytest.raises(ValueError, match="pointing zenith"):
        _ = table.pointing_ze
    with pytest.raises(ValueError, match="pointing azimuth"):
        _ = table.pointing_az
