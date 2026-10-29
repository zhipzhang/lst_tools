from types import SimpleNamespace

import astropy.units as u
import pytest

from lst_tools.dl2 import LSTDL2MCTable
from lst_tools.dl2.dl2table_base import LSTDL2TableBase

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


@pytest.mark.parametrize(
    ("viewcone_min", "viewcone_max", "expected_min", "expected_max"),
    [
        (0.3999, 0.4001, 0.0, 0.0),
        (0.0, 0.2, 0.0, 0.2),
    ],
)
def test_viewcone_selection(monkeypatch, viewcone_min, viewcone_max, expected_min, expected_max):
    simu_config = SimpleNamespace(
        n_showers=100,
        shower_reuse=10,
        energy_range_min=0.01 * u.TeV,
        energy_range_max=100 * u.TeV,
        max_scatter_range=1000 * u.m,
        spectral_index=-2.0,
        min_viewcone_radius=viewcone_min * u.deg,
        max_viewcone_radius=viewcone_max * u.deg,
    )
    monkeypatch.setattr(
        LSTDL2TableBase,
        "__init__",
        lambda self, file_name: setattr(self, "file_name", str(file_name)),
    )
    monkeypatch.setattr("lst_tools.dl2.dl2mctable.read_simu_info_merged_hdf5", lambda _: simu_config)

    table = LSTDL2MCTable("dummy.h5")

    assert table.sim_info.viewcone_min == expected_min * u.deg
    assert table.sim_info.viewcone_max == expected_max * u.deg
