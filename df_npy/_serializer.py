from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ._arrays import prepare_writable_array
from ._axis import deserialise_axis, serialise_axis
from ._constants import ARRAY_ORDER_FORTRAN, STRING_MISSING_VALUE_SENTINEL, MetadataKey
from ._dtypes import extract_dtype_plan, is_string_dtype, restore_column_dtypes
from ._json import json_default
from ._paths import ensure_npy_path, metadata_file_from_npy_file


class NpySerializer:
    @classmethod
    def get_columns(cls, file_path: Path | str) -> set[str]:
        """Return serialized DataFrame columns as a set of strings."""
        path = ensure_npy_path(file_path, for_write=False)
        logger.info(f"Reading columns metadata from {path}")
        if not path.exists():
            raise FileNotFoundError(path)

        metadata_path = metadata_file_from_npy_file(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if MetadataKey.COLUMNS.value not in metadata:
            raise ValueError("Metadata is missing required key: columns")

        columns = deserialise_axis(metadata[MetadataKey.COLUMNS.value])
        return {str(column) for column in columns}

    @classmethod
    def to_npy(cls, df: pd.DataFrame, file_path: Path | str) -> None:
        path = ensure_npy_path(file_path, for_write=True)
        if not df.columns.is_unique:
            raise ValueError("Columns must be unique for identifier-based subsetting.")

        logger.info(f"Serializing DataFrame to {path}")
        logger.debug("Extracting dtype from DataFrame")
        plan = extract_dtype_plan(df)

        logger.debug("Converting DataFrame to Fortran-ordered NumPy array")
        np_array = prepare_writable_array(df, plan)

        logger.debug("Creating metadata for DataFrame serialization")
        metadata = {
            MetadataKey.COLUMNS.value: serialise_axis(df.columns),
            MetadataKey.INDEX.value: serialise_axis(df.index),
            MetadataKey.DTYPE.value: str(plan.representative_dtype),
            MetadataKey.STORAGE_DTYPE.value: str(np_array.dtype),
            MetadataKey.COLUMN_DTYPES.value: plan.column_dtypes,
            MetadataKey.SHAPE.value: list(np_array.shape),
            MetadataKey.ORDER.value: ARRAY_ORDER_FORTRAN,
        }

        logger.debug("Saving array")
        np.save(path, np_array, allow_pickle=False)

        logger.debug("Saving metadata")
        metadata_path = metadata_file_from_npy_file(path)
        metadata_path.write_text(
            json.dumps(metadata, indent=4, default=json_default),
            encoding="utf-8",
        )
        logger.success(f"DataFrame serialized to {path}.")

    @classmethod
    def from_npy(
        cls,
        file_path: Path | str,
        *,
        identifiers: list[str] | None = None,
    ) -> pd.DataFrame:
        path = ensure_npy_path(file_path, for_write=False)
        logger.info(f"Deserializing DataFrame from {path}")
        if not path.exists():
            raise FileNotFoundError(path)

        logger.debug("Loading metadata from JSON file")
        metadata_path = metadata_file_from_npy_file(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if MetadataKey.COLUMN_DTYPES.value not in metadata:
            raise ValueError("Metadata is missing required key: column_dtypes")
        column_dtypes = metadata[MetadataKey.COLUMN_DTYPES.value]

        logger.debug("Loading NumPy array from file")
        np_array = np.load(path, allow_pickle=False, mmap_mode="r")
        if not np_array.flags["F_CONTIGUOUS"]:
            raise ValueError("Stored array is not Fortran contiguous as expected.")

        logger.debug("Deserializing index")
        index = deserialise_axis(metadata[MetadataKey.INDEX.value])

        logger.debug("Deserializing columns")
        columns = deserialise_axis(metadata[MetadataKey.COLUMNS.value])

        if identifiers is not None:
            logger.debug("Subsetting columns based on provided identifiers")
            positions = {column: position for position, column in enumerate(columns)}
            try:
                column_indices = [positions[column] for column in identifiers]
            except KeyError as exc:
                raise KeyError(f"Identifier not found in columns: {exc}") from None
            np_array = np_array[:, np.asarray(column_indices, dtype=np.int64)]
            columns = pd.Index(identifiers, name=columns.name)
            column_dtypes = [column_dtypes[index] for index in column_indices]
        else:
            _ = np_array[...]

        logger.debug("Creating DataFrame from NumPy array, columns, and index")
        df = pd.DataFrame(np_array, columns=columns, index=index, copy=False)

        if is_string_dtype(metadata.get(MetadataKey.DTYPE.value)):
            logger.debug("Replacing sentinel values with NaN for string/object dtype")
            df.replace(STRING_MISSING_VALUE_SENTINEL, np.nan, inplace=True)

        logger.debug("Restoring per-column dtypes from metadata")
        df = restore_column_dtypes(df, column_dtypes)

        logger.success(f"DataFrame deserialized from {path}.")
        logger.debug(f"Deserialized DataFrame shape: {df.shape}")
        return df
