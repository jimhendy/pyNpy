import json

import numpy as np
import pandas as pd
import pytest

from df_npy._arrays import prepare_writable_array
from df_npy._axis import _extract_time_unit, deserialise_axis, serialise_axis
from df_npy._constants import STRING_MISSING_VALUE_SENTINEL, MetadataKey
from df_npy._dtypes import DtypePlan, extract_dtype_plan, restore_column_dtypes
from df_npy._paths import ensure_npy_path, metadata_file_from_npy_file
from df_npy._serializer import NpySerializer


def test_extract_dtype_plan_mixed_numeric_tracks_column_dtypes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})

    plan = extract_dtype_plan(df)

    assert plan.representative_dtype == "float64"
    assert plan.mixed_numeric is True
    assert plan.column_dtypes == ["int64", "float64"]


def test_extract_dtype_plan_object_values_are_rejected():
    df = pd.DataFrame({"a": [1, "two", 3.0], "b": ["x", "y", "z"]}, dtype=object)

    with pytest.raises(
        ValueError, match="Pickle-backed object serialization is disabled"
    ):
        extract_dtype_plan(df)


def test_extract_dtype_plan_rejects_mixed_string_and_numeric_dtypes():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})

    with pytest.raises(ValueError, match="single-dtype frames"):
        extract_dtype_plan(df)


def test_prepare_writable_array_for_strings_replaces_missing_with_sentinel():
    df = pd.DataFrame({"a": ["x", None, "z"]}, dtype=object)
    plan = DtypePlan(
        representative_dtype="object",
        mixed_numeric=False,
        column_dtypes=["object"],
    )

    arr = prepare_writable_array(df, plan)

    assert arr.flags["F_CONTIGUOUS"]
    assert arr.dtype.kind == "U"
    assert STRING_MISSING_VALUE_SENTINEL in arr


def test_restore_column_dtypes_recasts_from_storage_values():
    restored_source = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.1, 2.2, 3.3],
            "c": [True, False, True],
        },
    )
    column_dtypes = ["int64", "float64", "bool"]

    restored = restore_column_dtypes(restored_source, column_dtypes)

    assert str(restored["a"].dtype) == "int64"
    assert str(restored["b"].dtype) == "float64"
    assert str(restored["c"].dtype) == "bool"


def test_extract_time_unit_handles_tz_datetime_dtype():
    assert _extract_time_unit("datetime64[us, UTC]") == "us"
    assert _extract_time_unit(None) == "ns"


def test_axis_serialise_deserialise_roundtrip_tz_datetime_index():
    index = pd.date_range("2023-01-01", periods=3, tz="Europe/London")

    metadata = serialise_axis(index)
    restored = deserialise_axis(metadata)

    pd.testing.assert_index_equal(index, restored)


def test_ensure_npy_path_and_metadata_path(tmp_path):
    base = tmp_path / "data" / "example"

    npy_path = ensure_npy_path(base, for_write=True)
    metadata_path = metadata_file_from_npy_file(npy_path)

    assert npy_path.suffix == ".npy"
    assert npy_path.parent.exists()
    assert metadata_path.suffix == ".json"


def test_restore_column_dtypes_rejects_mismatched_dtype_list_length():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})

    with pytest.raises(ValueError, match="length does not match"):
        restore_column_dtypes(df, ["float64"])


def test_restore_column_dtypes_preserves_unsigned_nullable_integer():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

    restored = restore_column_dtypes(df, ["uint64"])

    assert str(restored["a"].dtype) == "UInt64"


def test_restore_column_dtypes_is_noop_when_dtypes_already_match():
    df = pd.DataFrame({"a": [1.1, 2.2], "b": [3.3, 4.4]}, dtype="float64")

    restored = restore_column_dtypes(df, ["float64", "float64"])

    assert restored is df


def test_restore_column_dtypes_groups_same_target_dtype_columns():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [1.1, 2.2, 3.3],
        },
    )

    restored = restore_column_dtypes(df, ["int64", "int64", "float64"])

    assert str(restored["a"].dtype) == "int64"
    assert str(restored["b"].dtype) == "int64"
    assert str(restored["c"].dtype) == "float64"


def test_extract_dtype_plan_handles_non_string_column_label_collision():
    df = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=pd.Index([1, "1"]))

    plan = extract_dtype_plan(df)

    assert plan.column_dtypes == ["int64", "int64"]


def test_from_npy_raises_when_column_dtypes_missing(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.npy"
    NpySerializer.to_npy(df, file_path)

    metadata_path = file_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop(MetadataKey.COLUMN_DTYPES.value)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="column_dtypes"):
        NpySerializer.from_npy(file_path)
