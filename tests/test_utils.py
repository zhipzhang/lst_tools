import numpy as np
from astropy import units as u
from astropy.time import Time

from lst_tools.utils import altaz_to_icrs


def test_same_nparray_quantities():
    time_unix = np.array([1.7e9, 1.7e9 + 100, 1.7e9 + 200])
    alt_rad = np.array([1.0, 1.1, 1.2])
    az_rad = np.array([0.5, 0.6, 0.7])

    coord_from_arrays = altaz_to_icrs(time_unix, alt_rad, az_rad)

    coord_from_quantities = altaz_to_icrs(
        Time(time_unix, format="unix"),
        alt_rad * u.rad,
        az_rad * u.rad,
    )

    assert u.allclose(coord_from_arrays.ra, coord_from_quantities.ra)
    assert u.allclose(coord_from_arrays.dec, coord_from_quantities.dec)
