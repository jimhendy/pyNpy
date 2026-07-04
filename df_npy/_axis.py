from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd

from ._constants import DEFAULT_TIME_UNIT, AxisMetaKey, AxisType


def _extract_time_unit(dtype_name: str | None, default: str = DEFAULT_TIME_UNIT) -> str:
    if not dtype_name:
        return default
    match = re.search(r"\[(\w+)", dtype_name)
    if match:
        return match.group(1)
    return default


def serialise_axis(index: pd.Index | pd.MultiIndex) -> dict[str, Any]:
    if isinstance(index, pd.MultiIndex):
        return {
            AxisMetaKey.TYPE.value: AxisType.MULTIINDEX.value,
            AxisMetaKey.NAMES.value: list(index.names),
            AxisMetaKey.NLEVELS.value: index.nlevels,
            AxisMetaKey.LEVELS.value: [serialise_axis(level) for level in index.levels],
            AxisMetaKey.CODES.value: [codes.tolist() for codes in index.codes],
            AxisMetaKey.SORTORDER.value: index.sortorder,
        }

    payload: dict[str, Any] = {
        AxisMetaKey.TYPE.value: type(index).__name__,
        AxisMetaKey.NAME.value: index.name,
    }

    if isinstance(index, pd.RangeIndex):
        payload.update(
            {
                AxisMetaKey.RANGE.value: True,
                AxisMetaKey.START.value: int(index.start),
                AxisMetaKey.STOP.value: int(index.stop),
                AxisMetaKey.STEP.value: int(index.step),
            },
        )
        return payload

    if isinstance(index, pd.DatetimeIndex):
        payload.update(
            {
                AxisMetaKey.DATETIME.value: True,
                AxisMetaKey.DTYPE.value: index.dtype.name,
                AxisMetaKey.TZ.value: str(index.tz) if index.tz is not None else None,
                AxisMetaKey.FREQ.value: index.freqstr,
                AxisMetaKey.VALUES_I8.value: index.asi8.tolist(),
            },
        )
        return payload

    if isinstance(index, pd.TimedeltaIndex):
        payload.update(
            {
                AxisMetaKey.TIMEDELTA.value: True,
                AxisMetaKey.DTYPE.value: index.dtype.name,
                AxisMetaKey.FREQ.value: index.freqstr,
                AxisMetaKey.VALUES_I8.value: index.asi8.tolist(),
            },
        )
        return payload

    payload.update(
        {
            AxisMetaKey.VALUES.value: index.tolist(),
            AxisMetaKey.DTYPE.value: str(getattr(index, "dtype", "object")),
        },
    )
    return payload


def deserialise_axis(metadata: dict[str, Any]) -> pd.Index | pd.MultiIndex:
    axis_type = metadata.get(AxisMetaKey.TYPE.value)

    if axis_type == AxisType.MULTIINDEX.value:
        levels = [
            deserialise_axis(level) for level in metadata[AxisMetaKey.LEVELS.value]
        ]
        return pd.MultiIndex(
            levels=levels,
            codes=metadata[AxisMetaKey.CODES.value],
            names=metadata.get(AxisMetaKey.NAMES.value),
            sortorder=metadata.get(AxisMetaKey.SORTORDER.value),
        )

    if metadata.get(AxisMetaKey.RANGE.value):
        return pd.RangeIndex(
            start=metadata[AxisMetaKey.START.value],
            stop=metadata[AxisMetaKey.STOP.value],
            step=metadata[AxisMetaKey.STEP.value],
            name=metadata.get(AxisMetaKey.NAME.value),
        )

    if metadata.get(AxisMetaKey.DATETIME.value):
        unit = _extract_time_unit(metadata.get(AxisMetaKey.DTYPE.value))
        values = metadata.get(AxisMetaKey.VALUES_I8.value, [])
        tz = metadata.get(AxisMetaKey.TZ.value)

        if tz:
            idx = pd.to_datetime(values, unit=unit, utc=True)
            idx = pd.DatetimeIndex(
                idx,
                name=metadata.get(AxisMetaKey.NAME.value),
            ).tz_convert(tz)
        else:
            idx = pd.DatetimeIndex(
                pd.to_datetime(values, unit=unit),
                name=metadata.get(AxisMetaKey.NAME.value),
            )

        if freq := metadata.get(AxisMetaKey.FREQ.value):
            with suppress(ValueError):
                idx = pd.DatetimeIndex(idx, name=idx.name, freq=freq)
        return idx

    if metadata.get(AxisMetaKey.TIMEDELTA.value):
        unit = _extract_time_unit(metadata.get(AxisMetaKey.DTYPE.value))
        values_i8 = metadata.get(AxisMetaKey.VALUES_I8.value, [])
        td_arr = np.array(values_i8, dtype=f"timedelta64[{unit}]")
        idx = pd.TimedeltaIndex(
            td_arr,
            name=metadata.get(AxisMetaKey.NAME.value),
        )
        if freq := metadata.get(AxisMetaKey.FREQ.value):
            with suppress(ValueError):
                idx = pd.TimedeltaIndex(idx, name=idx.name, freq=freq)
        return idx

    return pd.Index(
        metadata.get(AxisMetaKey.VALUES.value, []),
        name=metadata.get(AxisMetaKey.NAME.value),
    )
