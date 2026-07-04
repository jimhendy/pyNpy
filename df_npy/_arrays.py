from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from ._constants import (
    NUMPY_DTYPE_FLOAT64,
    NUMPY_DTYPE_UNICODE,
    STRING_MISSING_VALUE_SENTINEL,
)
from ._dtypes import DtypePlan, is_string_dtype


def prepare_writable_array(df: pd.DataFrame, plan: DtypePlan) -> np.ndarray:
    if is_string_dtype(plan.representative_dtype):
        array = (
            df.astype(object)
            .where(df.notna(), other=STRING_MISSING_VALUE_SENTINEL)
            .to_numpy(dtype=NUMPY_DTYPE_UNICODE, copy=False)
        )
        return np.asfortranarray(array)

    if plan.mixed_numeric:
        return np.asfortranarray(df.to_numpy(dtype=NUMPY_DTYPE_FLOAT64, copy=False))

    return np.asfortranarray(df.to_numpy(dtype=plan.representative_dtype, copy=False))
