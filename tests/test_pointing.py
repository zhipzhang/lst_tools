import numpy as np
import pytest
from astropy import units as u

from lst_tools.pointing import Pointing


@pytest.fixture
def pointing():
    return Pointing(
        pointing_alt=np.array([1.0, 1.01, 0.99]),
        pointing_az=np.array([0.5, 0.51, 0.49]),
        trigger_time=np.array([1.7e9, 1.7e9 + 60, 1.7e9 + 120]),
    )


def test_altaz_preserves_input_pointing(pointing):
    np.testing.assert_allclose(pointing.altaz.alt.to_value(u.rad), [1.0, 1.01, 0.99])
    np.testing.assert_allclose(pointing.altaz.az.to_value(u.rad), [0.5, 0.51, 0.49])


def test_mean_pointing_of_fixed_direction():
    pointing = Pointing(
        pointing_alt=np.full(3, 1.0),
        pointing_az=np.full(3, 0.5),
        trigger_time=np.full(3, 1.7e9),
    )

    expected = pointing.icrs[0]

    assert pointing.pointing_ra.to_value(u.deg) == pytest.approx(expected.ra.to_value(u.deg))
    assert pointing.pointing_dec.to_value(u.deg) == pytest.approx(expected.dec.to_value(u.deg))
