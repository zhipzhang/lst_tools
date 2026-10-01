"""Adapters from LST-specific DL2 data to generic background event samples."""

from collections.abc import Callable

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord, SkyOffsetFrame

from lst_tools.dl2 import LSTDL2EventTable

from ..location import LST_LOCATION
from ..utils import altaz_to_icrs
from .events import SkyOffsetEvents


def sky_offset_events_from_lstdl2(
    dl2: LSTDL2EventTable,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    location: EarthLocation = LST_LOCATION,
) -> SkyOffsetEvents:
    """Convert one LST DL2 event table into center-free sky-offset events.

    Reconstructed directions are interpreted as Alt/Az coordinates in
    radians, trigger times as Unix seconds, and reconstructed energies as
    TeV. All directions are transformed into one sky-offset frame centered
    on the run pointing given by ``pointing_ra`` and ``pointing_dec``. The
    center is used during conversion but is not retained in the result.

    Parameters
    ----------
    dl2
        Object exposing a DL2 event DataFrame as ``data``, its effective
        livetime in seconds as ``t_eff``, and the run pointing as
        ``pointing_ra``/``pointing_dec``.
        :class:`~lst_tools.dl2.LSTDL2EventTable` satisfies this interface.
    event_filter
        Optional callable returning the rows to retain. The complete input
        effective livetime is preserved after filtering.
    location
        Observatory location used for the Alt/Az-to-ICRS transformation.
    """
    try:
        data = dl2.data
        livetime = u.Quantity(dl2.t_eff, u.s)
        pointing = SkyCoord(ra=dl2.pointing_ra, dec=dl2.pointing_dec, frame="icrs")
    except AttributeError as error:
        raise TypeError("dl2 must expose 'data', 't_eff', and 'pointing_ra'/'pointing_dec' attributes") from error

    if not isinstance(data, pd.DataFrame):
        raise TypeError("dl2.data must be a pandas DataFrame")

    selected = event_filter(data) if event_filter is not None else data

    if selected.empty:
        return SkyOffsetEvents(livetime=livetime)

    event_time = selected["trigger_time"].to_numpy(dtype=float)
    reconstructed = altaz_to_icrs(
        event_time,
        selected["reco_alt"].to_numpy(dtype=float),
        selected["reco_az"].to_numpy(dtype=float),
        location=location,
    )

    offsets = reconstructed.transform_to(SkyOffsetFrame(origin=pointing))
    return SkyOffsetEvents(
        x=offsets.lon,
        y=offsets.lat,
        energy=selected["reco_energy"].to_numpy(dtype=float) * u.TeV,
        livetime=livetime,
    )
