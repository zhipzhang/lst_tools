"""Adapters from LST data to generic background event samples."""

from collections.abc import Callable
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from gammapy.data import EventList
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame

from lst_tools.dl2 import LSTDL2EventTable, LSTDL2MCTable
from lst_tools.pointing import Pointing

from ..location import LST_LOCATION
from ..utils import altaz_to_icrs
from .events import ObservationFrame, SkyOffsetEvents

_DEFAULT_MC_TIME = Time("2026-01-01")


def sky_offset_events_from_dl3(
    events: EventList,
    location: EarthLocation = LST_LOCATION,
    frame: Literal["icrs", "altaz"] = "icrs",
) -> SkyOffsetEvents:
    """Convert all DL3 events into sky-offset events without source masking."""
    if frame != "icrs":
        raise NotImplementedError("altaz frame not yet supported")

    pointing = events.pointing_radec
    fov_frame = FoVICRSFrame(origin=pointing)
    offsets = events.radec.transform_to(fov_frame)
    mean_time = events.time.mean()
    pointing_altaz = pointing.transform_to(AltAz(obstime=mean_time, location=location))
    observation_frame = ObservationFrame(
        fov_frame=fov_frame,
        mean_time=mean_time,
        pointing_alt=pointing_altaz.alt,
        pointing_az=pointing_altaz.az,
    )
    return SkyOffsetEvents(
        x=offsets.fov_lon,
        y=offsets.fov_lat,
        energy=events.energy,
        livetime=events.observation_live_time_duration,
        frame=observation_frame,
    )


def _mean_dl2_pointing(data: pd.DataFrame) -> tuple[Time, u.Quantity, u.Quantity]:
    trigger_time = data["trigger_time"].to_numpy(dtype=float)
    mean_time = Time(np.mean(trigger_time), format="unix")
    pointing_alt = u.Quantity(data["alt_tel"].to_numpy(dtype=float), u.rad)
    pointing_az = u.Quantity(data["az_tel"].to_numpy(dtype=float), u.rad)
    mean_az = np.arctan2(np.sin(pointing_az).mean(), np.cos(pointing_az).mean())
    return mean_time, pointing_alt.mean(), mean_az.to(u.deg)


def sky_offset_events_from_lstdl2(
    dl2: LSTDL2EventTable,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    location: EarthLocation = LST_LOCATION,
    frame: Literal["icrs", "altaz"] = "icrs",
) -> SkyOffsetEvents:
    """Convert one LST DL2 observation into center-free sky-offset events."""
    try:
        data = dl2.data
        livetime = u.Quantity(dl2.t_eff, u.s)
    except AttributeError as error:
        raise TypeError("dl2 must expose 'data' and 't_eff' attributes") from error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dl2.data must be a pandas DataFrame")

    mean_time, pointing_alt, pointing_az = _mean_dl2_pointing(data)
    selected = event_filter(data) if event_filter is not None else data
    event_time = selected["trigger_time"].to_numpy(dtype=float)

    if frame == "icrs":
        pointing = SkyCoord(ra=dl2.pointing_ra, dec=dl2.pointing_dec, frame="icrs")
        fov_frame = FoVICRSFrame(origin=pointing)
        if not selected.empty:
            reconstructed = altaz_to_icrs(
                event_time,
                selected["reco_alt"].to_numpy(dtype=float),
                selected["reco_az"].to_numpy(dtype=float),
                location=location,
            )
            offsets = reconstructed.transform_to(fov_frame)
    elif frame == "altaz":
        altaz_frame = Pointing(
            pointing_alt=selected["alt_tel"],
            pointing_az=selected["az_tel"],
            trigger_time=event_time,
        ).altaz
        fov_frame = FoVAltAzFrame(origin=altaz_frame, location=location)
        if not selected.empty:
            reconstructed = SkyCoord(
                alt=selected["reco_alt"].to_numpy(dtype=float) * u.rad,
                az=selected["reco_az"].to_numpy(dtype=float) * u.rad,
                frame=altaz_frame,
            )
            offsets = reconstructed.transform_to(fov_frame)
    else:
        raise ValueError("frame must be 'icrs' or 'altaz'")

    observation_frame = ObservationFrame(
        fov_frame=fov_frame,
        mean_time=mean_time,
        pointing_alt=pointing_alt,
        pointing_az=pointing_az,
    )
    if selected.empty:
        return SkyOffsetEvents(livetime=livetime, frame=observation_frame)

    return SkyOffsetEvents(
        x=offsets.fov_lon,
        y=offsets.fov_lat,
        energy=selected["reco_energy"].to_numpy(dtype=float) * u.TeV,
        livetime=livetime,
        frame=observation_frame,
    )


def sky_offset_events_from_lstmcdl2(
    dl2: LSTDL2MCTable,
    time: Time = _DEFAULT_MC_TIME,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    use_mc_energy: bool = False,
    use_mc_direction: bool = True,
) -> SkyOffsetEvents:
    """Convert MC DL2 events using the fixed pointing encoded in the table."""
    pointing = Pointing(
        pointing_alt=90 * u.deg - dl2.pointing_ze,
        pointing_az=dl2.pointing_az,
        trigger_time=time,
    ).altaz
    fov_frame = FoVAltAzFrame(origin=pointing, location=LST_LOCATION)
    observation_frame = ObservationFrame(
        fov_frame=fov_frame,
        mean_time=time,
        pointing_alt=pointing.alt,
        pointing_az=pointing.az,
    )
    selected = event_filter(dl2.data) if event_filter is not None else dl2.data
    if selected.empty:
        return SkyOffsetEvents(livetime=50 * u.h, frame=observation_frame)

    direction_prefix = "mc" if use_mc_direction else "reco"
    directions = SkyCoord(
        alt=selected[f"{direction_prefix}_alt"].to_numpy(dtype=float) * u.rad,
        az=selected[f"{direction_prefix}_az"].to_numpy(dtype=float) * u.rad,
        frame=AltAz(obstime=time, location=LST_LOCATION),
    )
    offsets = directions.transform_to(fov_frame)
    energy_column = "mc_energy" if use_mc_energy else "reco_energy"
    return SkyOffsetEvents(
        x=offsets.fov_lon,
        y=offsets.fov_lat,
        energy=selected[energy_column].to_numpy(dtype=float) * u.TeV,
        livetime=50 * u.h,
        frame=observation_frame,
    )
