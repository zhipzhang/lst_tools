import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time

from .location import LST_LOCATION

DEFAULT_TIME_RESOLUTION = 100 * u.s


def altaz_to_icrs(
    time: np.ndarray | Time,
    alt: np.ndarray | u.Quantity,
    az: np.ndarray | u.Quantity,
    location: EarthLocation = LST_LOCATION,
    *,
    time_resolution: u.Quantity = DEFAULT_TIME_RESOLUTION,
) -> SkyCoord:
    """Transform event-wise AltAz coordinates to ICRS."""
    if isinstance(time, np.ndarray):
        time = Time(time, format="unix")
    if not isinstance(alt, u.Quantity):
        alt = alt * u.Unit("rad")
    if not isinstance(az, u.Quantity):
        az = az * u.Unit("rad")

    altaz = SkyCoord(
        alt=alt,
        az=az,
        frame=AltAz(
            obstime=time,
            location=location,
        ),
    )

    with erfa_astrom.set(ErfaAstromInterpolator(time_resolution=time_resolution)):
        return altaz.icrs


def mean_direction(coordinates: SkyCoord) -> SkyCoord:
    """Return the spherical mean of a set of ICRS directions."""
    mean_x, mean_y, mean_z = coordinates.cartesian.xyz.mean(axis=1)
    mean = SkyCoord(
        x=mean_x,
        y=mean_y,
        z=mean_z,
        representation_type="cartesian",
        frame="icrs",
    )
    return SkyCoord(ra=mean.spherical.lon, dec=mean.spherical.lat, frame="icrs")
