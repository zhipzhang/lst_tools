import astropy.units as u
from astropy.coordinates import AltAz, Latitude, Longitude, SkyCoord
from astropy.time import Time
from numpy.typing import ArrayLike

from .location import LST_LOCATION
from .utils import altaz_to_icrs, mean_direction


class Pointing:
    pointing_alt: u.Quantity
    pointing_az: u.Quantity
    trigger_time: Time

    def __init__(
        self,
        pointing_alt: ArrayLike | u.Quantity,
        pointing_az: ArrayLike | u.Quantity,
        trigger_time: ArrayLike | Time,
    ) -> None:
        if isinstance(pointing_alt, u.Quantity):
            self.pointing_alt = pointing_alt
        else:
            self.pointing_alt = u.Quantity(pointing_alt, "rad")

        if isinstance(pointing_az, u.Quantity):
            self.pointing_az = pointing_az
        else:
            self.pointing_az = u.Quantity(pointing_az, "rad")

        if isinstance(trigger_time, Time):
            self.trigger_time = trigger_time
        else:
            self.trigger_time = Time(trigger_time, format="unix")

    @property
    def altaz(self) -> SkyCoord:
        frame = AltAz(
            obstime=self.trigger_time,
            location=LST_LOCATION,
        )
        return SkyCoord(
            alt=self.pointing_alt,
            az=self.pointing_az,
            frame=frame,
        )

    @property
    def icrs(self):
        return altaz_to_icrs(
            self.trigger_time,
            self.pointing_alt,
            self.pointing_az,
            location=LST_LOCATION,
        )

    @property
    def pointing_ra(self):
        """Return the mean pointing right ascension.

        Returns
        -------
        Longitude
            Mean right ascension of the pointing direction.
        """
        return mean_direction(self.icrs).ra

    @property
    def pointing_dec(self):
        """Return the mean pointing declination.

        Returns
        -------
        Latitude
            Mean declination of the pointing direction.
        """
        return mean_direction(self.icrs).dec
