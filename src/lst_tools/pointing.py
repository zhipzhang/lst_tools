from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from numpy.typing import NDArray

from .location import LST_LOCATION
from .utils import altaz_to_icrs, mean_direction


@dataclass
class Pointing:
    pointing_alt: NDArray | u.Quantity
    pointing_az: NDArray | u.Quantity
    trigger_time: NDArray | Time

    def __post_init__(self):
        if not isinstance(self.pointing_alt, u.Quantity):
            self.pointing_alt = self.pointing_alt * u.Unit("rad")
        if not isinstance(self.pointing_az, u.Quantity):
            self.pointing_az = self.pointing_az * u.Unit("rad")
        if not isinstance(self.trigger_time, Time):
            self.trigger_time = Time(self.trigger_time, format="unix")

    @property
    def altaz(self):
        frame = AltAz(obstime=self.trigger_time, location=LST_LOCATION)
        return SkyCoord(self.pointing_alt, self.pointing_az, frame=frame)

    @property
    def icrs(self):
        return altaz_to_icrs(
            self.trigger_time,
            self.pointing_alt.to_value("rad"),
            self.pointing_az.to_value("rad"),
            location=LST_LOCATION,
        )

    @property
    def pointing_ra(self):
        mean_pointing = mean_direction(self.icrs)
        return mean_pointing.icrs.ra

    @property
    def pointing_dec(self):
        mean_pointing = mean_direction(self.icrs)
        return mean_pointing.ircs.dec
