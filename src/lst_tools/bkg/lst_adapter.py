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

from lst_tools.dl2 import LSTObservedDL2Table, LSTSimulatedDL2Table

from ..location import LST_LOCATION
from ..utils import altaz_to_icrs, icrs_to_altaz
from .events import ObservationFrame, SkyOffsetEvents

_DEFAULT_MC_TIME = Time("2026-01-01")


def sky_offset_events_from_dl3(
    events: EventList,
    location: EarthLocation = LST_LOCATION,
    frame: Literal["icrs", "altaz"] = "icrs",
) -> SkyOffsetEvents:
    """Convert all DL3 events into sky-offset events without source masking."""
    pointing = events.pointing_radec
    mean_time = events.time.mean()
    if frame == "icrs":
        fov_frame = FoVICRSFrame(origin=pointing)
        offsets = events.radec.transform_to(fov_frame)
    elif frame == "altaz":
        event_pointing = icrs_to_altaz(events.time, pointing.ra, pointing.dec, location=location)
        fov_frame = FoVAltAzFrame(origin=event_pointing, location=location)
        reconstructed = icrs_to_altaz(events.time, events.radec.ra, events.radec.dec, location=location)
        offsets = reconstructed.transform_to(fov_frame)
    else:
        raise ValueError("frame must be 'icrs' or 'altaz'")

    pointing_altaz = icrs_to_altaz(mean_time, pointing.ra, pointing.dec, location=location)
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
        livetime=events.observation_time_duration,
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
    dl2: LSTObservedDL2Table,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    location: EarthLocation = LST_LOCATION,
    frame: Literal["icrs", "altaz"] = "icrs",
) -> SkyOffsetEvents:
    """Convert one LST DL2 observation into sky-offset events.

    With ``frame="icrs"``, all events are centered on the fixed run pointing.
    With ``frame="altaz"``, each event is centered on its own telescope
    pointing at its trigger time. Filtering changes the selected events but
    preserves the observation's full elapsed time, including dead time.
    """
    try:
        data = dl2.events
        livetime = u.Quantity(dl2.elapsed_time, u.s)
    except AttributeError as error:
        raise TypeError("dl2 must expose 'events' and 'elapsed_time' attributes") from error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dl2.events must be a pandas DataFrame")

    mean_time, pointing_alt, pointing_az = _mean_dl2_pointing(data)
    selected = event_filter(data) if event_filter is not None else data
    event_time = Time(selected["trigger_time"].to_numpy(dtype=float), format="unix")

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
        altaz_frame = AltAz(obstime=event_time, location=location)
        event_pointing = SkyCoord(
            alt=selected["alt_tel"].to_numpy(dtype=float) * u.rad,
            az=selected["az_tel"].to_numpy(dtype=float) * u.rad,
            frame=altaz_frame,
        )
        fov_frame = FoVAltAzFrame(origin=event_pointing, location=location)
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
    dl2: LSTSimulatedDL2Table,
    time: Time = _DEFAULT_MC_TIME,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    use_mc_energy: bool = False,
    use_mc_direction: bool = True,
) -> SkyOffsetEvents:
    """Convert simulated LST DL2 events into FoV-AltAz sky-offset events.

    The fixed pointing is derived from the zenith and azimuth encoded in the
    table. MC directions and reconstructed energies are used by default;
    ``use_mc_direction`` and ``use_mc_energy`` select those inputs separately.
    The simulated observation has a fixed livetime of 50 hours.
    """
    altaz_frame = AltAz(obstime=time, location=LST_LOCATION)
    pointing = SkyCoord(
        alt=90 * u.deg - dl2.pointing_ze,
        az=dl2.pointing_az,
        frame=altaz_frame,
    )
    fov_frame = FoVAltAzFrame(origin=pointing, location=LST_LOCATION)
    observation_frame = ObservationFrame(
        fov_frame=fov_frame,
        mean_time=time,
        pointing_alt=pointing.alt,
        pointing_az=pointing.az,
    )
    livetime = 50 * u.h
    selected = event_filter(dl2.events) if event_filter is not None else dl2.events
    if selected.empty:
        return SkyOffsetEvents(livetime=livetime, frame=observation_frame)

    direction_prefix = "mc" if use_mc_direction else "reco"
    directions = SkyCoord(
        alt=selected[f"{direction_prefix}_alt"].to_numpy(dtype=float) * u.rad,
        az=selected[f"{direction_prefix}_az"].to_numpy(dtype=float) * u.rad,
        frame=altaz_frame,
    )
    offsets = directions.transform_to(fov_frame)
    energy_column = "mc_energy" if use_mc_energy else "reco_energy"
    return SkyOffsetEvents(
        x=offsets.fov_lon,
        y=offsets.fov_lat,
        energy=selected[energy_column].to_numpy(dtype=float) * u.TeV,
        livetime=livetime,
        frame=observation_frame,
    )
