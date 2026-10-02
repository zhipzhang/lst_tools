"""Adapters from LST-specific DL2 data to generic background event samples."""

from collections.abc import Callable
from typing import Literal

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame

from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.pointing import Pointing

from ..location import LST_LOCATION
from ..utils import altaz_to_icrs
from .events import SkyOffsetEvents


def sky_offset_events_from_lstdl2(
    dl2: LSTDL2EventTable,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    location: EarthLocation = LST_LOCATION,
    frame: Literal["icrs", "altaz"] = "icrs",
) -> SkyOffsetEvents:
    """Convert one LST DL2 event table into center-free sky-offset events.

    Reconstructed directions are interpreted as Alt/Az coordinates in
    radians, trigger times as Unix seconds, and reconstructed energies as
    TeV. Directions are transformed into a field-of-view offset frame; the
    frame center is used during conversion but is not retained in the
    result. The FoV offset longitude increases opposite to the sky
    longitude, i.e. it is reversed with respect to
    `~astropy.coordinates.SkyOffsetFrame`.

    Parameters
    ----------
    dl2
        Object exposing a DL2 event DataFrame as ``data`` and its effective
        livetime in seconds as ``t_eff``.
        :class:`~lst_tools.dl2.LSTDL2EventTable` satisfies this interface.
    event_filter
        Optional callable returning the rows to retain. The complete input
        effective livetime is preserved after filtering.
    location
        Observatory location used for the coordinate transformations.
    frame
        Offset frame to build. With ``"icrs"`` (default), offsets are
        computed in a FoV-ICRS frame centered on the run pointing given by
        the ``pointing_ra``/``pointing_dec`` attributes. With ``"altaz"``,
        offsets are computed in a FoV-AltAz frame centered on the per-event
        pointing given by the ``pointing_alt``/``pointing_az`` columns (in
        radians) at each event's trigger time.
    """
    try:
        data = dl2.data
        livetime = u.Quantity(dl2.t_eff, u.s)
    except AttributeError as error:
        raise TypeError("dl2 must expose 'data' and 't_eff' attributes") from error

    if not isinstance(data, pd.DataFrame):
        raise TypeError("dl2.data must be a pandas DataFrame")

    selected = event_filter(data) if event_filter is not None else data

    if selected.empty:
        return SkyOffsetEvents(livetime=livetime)

    event_time = selected["trigger_time"].to_numpy(dtype=float)
    if frame == "icrs":
        pointing = SkyCoord(ra=dl2.pointing_ra, dec=dl2.pointing_dec, frame="icrs")
        reconstructed = altaz_to_icrs(
            event_time,
            selected["reco_alt"].to_numpy(dtype=float),
            selected["reco_az"].to_numpy(dtype=float),
            location=location,
        )

        offsets = reconstructed.transform_to(FoVICRSFrame(origin=pointing))
    elif frame == "altaz":
        altaz_frame = Pointing(
            pointing_alt=selected["pointing_alt"],
            pointing_az=selected["pointing_az"],
            trigger_time=event_time,
        ).altaz
        events = SkyCoord(
            alt=selected["reco_alt"].to_numpy(dtype=float) * u.Unit("rad"),
            az=selected["reco_az"].to_numpy(dtype=float) * u.Unit("rad"),
            frame=altaz_frame,
        )
        fov_frame = FoVAltAzFrame(
            origin=altaz_frame,
            location=location,
        )
        offsets = events.transform_to(fov_frame)

    return SkyOffsetEvents(
        x=offsets.fov_lon,
        y=offsets.fov_lat,
        energy=selected["reco_energy"].to_numpy(dtype=float) * u.TeV,
        livetime=livetime,
    )
