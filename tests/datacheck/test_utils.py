import pytest

from lst_tools.datacheck.utils import azimuth_angle_mean


def test_azimuth_angle_mean():
    azimuth_angles = [0.015, 359.99]
    result = azimuth_angle_mean(azimuth_angles)
    assert result == pytest.approx(0, abs=0.005)
