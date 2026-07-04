from __future__ import annotations

import numpy as np


def safe_str(obj: object) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def json_default(obj: object) -> bool | float | int | str:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)
    return safe_str(obj)
