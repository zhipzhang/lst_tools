"""Adapters from LST-specific DL2 data to generic background event samples."""

from collections.abc import Callable
from typing import Protocol

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation, SkyOffsetFrame

from ..location import LST_LOCATION
from ..utils import altaz_to_icrs, mean_direction
from .events import SkyOffsetEvents


class LSTDL2EventTableLike(Protocol):
    """Structural input required by :func:`sky_offset_events_from_lstdl2`."""

    data: pd.DataFrame
    t_eff: float | u.Quantity


REQUIRED_COLUMNS = ("reco_alt", "reco_az", "reco_energy", "alt_tel", "az_tel", "trigger_time")


def sky_offset_events_from_lstdl2(
    dl2: LSTDL2EventTableLike,
    event_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    *,
    location: EarthLocation = LST_LOCATION,
) -> SkyOffsetEvents:
    """Convert one LST DL2 event table into center-free sky-offset events.

    Reconstructed directions and telescope pointings are interpreted as
    Alt/Az coordinates in radians, trigger times as Unix seconds, and
    reconstructed energies as TeV. All directions are transformed into one
    sky-offset frame centered on the spherical mean telescope pointing. The
    center is used during conversion but is not retained in the result.

    Parameters
    ----------
    dl2
        Object exposing a DL2 event DataFrame as ``data`` and its effective
        livetime as ``t_eff``. :class:`~lst_tools.dl2.LSTDL2EventTable`
        satisfies this interface.
    event_filter
        Optional callable returning the rows to retain. The complete input
        effective livetime is preserved after filtering.
    location
        Observatory location used for the Alt/Az-to-ICRS transformation.
    """
    try:
        data = dl2.data
        livetime = dl2.t_eff
    except AttributeError as error:
        raise TypeError("dl2 must expose 'data' and 't_eff' attributes") from error

    if not isinstance(data, pd.DataFrame):
        raise TypeError("dl2.data must be a pandas DataFrame")

    selected = event_filter(data) if event_filter is not None else data
    if not isinstance(selected, pd.DataFrame):
        raise TypeError("event_filter must return a pandas DataFrame")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in selected.columns]
    if missing_columns:
        raise ValueError(f"DL2 events are missing required columns: {', '.join(missing_columns)}")

    if selected.empty:
        return SkyOffsetEvents(livetime=livetime)

    event_time = selected["trigger_time"].to_numpy(dtype=float)
    pointing = altaz_to_icrs(
        event_time,
        selected["alt_tel"].to_numpy(dtype=float),
        selected["az_tel"].to_numpy(dtype=float),
        location=location,
    )
    reconstructed = altaz_to_icrs(
        event_time,
        selected["reco_alt"].to_numpy(dtype=float),
        selected["reco_az"].to_numpy(dtype=float),
        location=location,
    )

    offsets = reconstructed.transform_to(SkyOffsetFrame(origin=mean_direction(pointing)))
    return SkyOffsetEvents(
        x=offsets.lon,
        y=offsets.lat,
        energy=selected["reco_energy"].to_numpy(dtype=float) * u.TeV,
        livetime=livetime,
    )
