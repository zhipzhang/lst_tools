import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.utils.coordinates import FoVICRSFrame

from lst_tools.bkg import ObservationFrame


@pytest.fixture
def observation_frame():
    origin = SkyCoord(ra=83.5, dec=20.3, unit=u.deg, frame="icrs")
    return ObservationFrame(
        fov_frame=FoVICRSFrame(origin=origin),
        mean_time=Time(1_700_000_000, format="unix"),
        pointing_alt=70 * u.deg,
        pointing_az=10 * u.deg,
    )
