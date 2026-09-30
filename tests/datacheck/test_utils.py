import numpy as np
import pandas as pd
import pytest

from lst_tools.datacheck.utils import (
    assign_zenith_bins,
    azimuth_angle_mean,
)


def test_azimuth_angle_mean():
    azimuth_angles = [0.015, 359.99]
    result = azimuth_angle_mean(azimuth_angles)
    assert result == pytest.approx(0, abs=0.005)


def test_assign_zenith_bins_has_unique_boundaries():
    angles = np.array([0.0, 19.999, 20.0, 39.999, 40.0, 70.0])
    statistics = pd.DataFrame({"mean_cos_zd": np.cos(np.radians(angles))})

    result = assign_zenith_bins(statistics, [0, 20, 40, 70])

    assert result["zenith_bin"].tolist() == [
        "zd_0_20",
        "zd_0_20",
        "zd_20_40",
        "zd_20_40",
        "zd_40_70",
        "zd_40_70",
    ]
    assert result["mean_zenith_angle"].to_numpy() == pytest.approx(angles)


@pytest.mark.parametrize(
    "edges",
    [[], [20], [0, 20, 20], [0, 30, 20], [-1, 20], [0, 91], [0, np.nan, 20]],
)
def test_assign_zenith_bins_rejects_invalid_edges(edges):
    statistics = pd.DataFrame({"mean_cos_zd": [1.0]})

    with pytest.raises(ValueError):
        assign_zenith_bins(statistics, edges)
