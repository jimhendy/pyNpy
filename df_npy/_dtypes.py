from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from numpy.typing import DTypeLike

from ._constants import (
    NUMPY_DTYPE_FLOAT64,
    PANDAS_BOOL_DTYPE,
    PANDAS_NULLABLE_BOOL_DTYPE,
    STRING_DTYPE_NAMES,
)


@dataclass(frozen=True)
class DtypePlan:
    representative_dtype: DTypeLike
    mixed_numeric: bool
    column_dtypes: list[str]


def is_string_dtype(dtype: object) -> bool:
    if dtype is None:
        return False
    try:
        return pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(
            dtype,
        )
    except (TypeError, ValueError):
        return str(dtype) in STRING_DTYPE_NAMES


def is_numeric_dtype(dtype: object) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(dtype)
    except (TypeError, ValueError):
        return False


def is_integer_dtype(dtype: object) -> bool:
    try:
        return pd.api.types.is_integer_dtype(dtype)
    except (TypeError, ValueError):
        return False


def is_float_dtype(dtype: object) -> bool:
    try:
        return pd.api.types.is_float_dtype(dtype)
    except (TypeError, ValueError):
        return False


def extract_dtype_plan(df: pd.DataFrame) -> DtypePlan:
    column_dtypes = [str(dtype) for dtype in df.dtypes]
    dtypes = df.dtypes
    n_distinct_dtypes = dtypes.nunique()

    if n_distinct_dtypes == 0:
        return DtypePlan(
            representative_dtype=NUMPY_DTYPE_FLOAT64,
            mixed_numeric=False,
            column_dtypes=column_dtypes,
        )

    if any(is_string_dtype(dt) for dt in dtypes):
        if n_distinct_dtypes > 1:
            msg = (
                f"DataFrame has {n_distinct_dtypes} distinct dtypes; "
                "only single-dtype frames are supported "
                "(string/object cannot be mixed)."
            )
            raise ValueError(
                msg,
            )
        non_null = df.stack().dropna()
        has_non_string = not non_null.map(lambda value: isinstance(value, str)).all()
        if has_non_string:
            msg = (
                "Pickle-backed object serialization is disabled; "
                "object/string frames must contain only string values "
                "and missing values."
            )
            raise ValueError(
                msg,
            )
        return DtypePlan(
            representative_dtype=dtypes.iloc[0],
            mixed_numeric=False,
            column_dtypes=column_dtypes,
        )

    if all(is_numeric_dtype(dt) for dt in dtypes):
        has_int = any(is_integer_dtype(dt) for dt in dtypes)
        has_float = any(is_float_dtype(dt) for dt in dtypes)
        if has_int and has_float:
            return DtypePlan(
                representative_dtype=NUMPY_DTYPE_FLOAT64,
                mixed_numeric=True,
                column_dtypes=column_dtypes,
            )
        if n_distinct_dtypes > 1:
            msg = (
                f"DataFrame has {n_distinct_dtypes} distinct numeric dtypes; "
                "only int+float mixing is supported."
            )
            raise ValueError(
                msg,
            )
        return DtypePlan(
            representative_dtype=dtypes.iloc[0],
            mixed_numeric=False,
            column_dtypes=column_dtypes,
        )

    if n_distinct_dtypes > 1:
        msg = (
            f"DataFrame has {n_distinct_dtypes} distinct dtypes; "
            "only single-dtype frames are supported."
        )
        raise ValueError(
            msg,
        )

    return DtypePlan(
        representative_dtype=dtypes.iloc[0],
        mixed_numeric=False,
        column_dtypes=column_dtypes,
    )


def _nullable_integer_dtype(dtype_name: str) -> str:
    match = re.fullmatch(r"(u?)int(8|16|32|64)", dtype_name)
    if not match:
        return "Int64"
    unsigned, bits = match.groups()
    prefix = "UInt" if unsigned else "Int"
    return f"{prefix}{bits}"


def restore_column_dtypes(df: pd.DataFrame, column_dtypes: list[str]) -> pd.DataFrame:
    if len(column_dtypes) != len(df.columns):
        raise ValueError(
            "column_dtypes length does not match DataFrame columns length.",
        )

    # Fast path: avoid per-column recasting when all dtypes already match.
    current_dtypes = [str(dtype) for dtype in df.dtypes]
    if current_dtypes == column_dtypes:
        return df

    grouped_columns: dict[str, list[object]] = {}
    for position, column in enumerate(df.columns):
        grouped_columns.setdefault(column_dtypes[position], []).append(column)

    for dtype_name, columns in grouped_columns.items():
        if dtype_name.startswith(("int", "uint")):
            na_columns = df[columns].isna().any(axis=0)
            with_na = [column for column, has_na in na_columns.items() if has_na]
            without_na = [column for column in columns if column not in with_na]

            if without_na:
                numeric = df[without_na].apply(pd.to_numeric, errors="raise")
                df[without_na] = numeric.astype(dtype_name)

            if with_na:
                nullable_dtype = _nullable_integer_dtype(dtype_name)
                numeric = df[with_na].apply(pd.to_numeric, errors="coerce")
                df[with_na] = numeric.astype(nullable_dtype)
        elif dtype_name.startswith("float"):
            numeric = df[columns].apply(pd.to_numeric, errors="coerce")
            df[columns] = numeric.astype(dtype_name)
        elif dtype_name == PANDAS_BOOL_DTYPE:
            df[columns] = df[columns].astype(PANDAS_BOOL_DTYPE)
        elif dtype_name == PANDAS_NULLABLE_BOOL_DTYPE:
            df[columns] = df[columns].astype(PANDAS_NULLABLE_BOOL_DTYPE)
        else:
            df[columns] = df[columns].astype(dtype_name)

    return df
