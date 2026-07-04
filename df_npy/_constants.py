from enum import StrEnum


class MetadataKey(StrEnum):
    COLUMNS = "columns"
    INDEX = "index"
    DTYPE = "dtype"
    STORAGE_DTYPE = "storage_dtype"
    COLUMN_DTYPES = "column_dtypes"
    SHAPE = "shape"
    ORDER = "order"


class AxisMetaKey(StrEnum):
    TYPE = "type"
    NAME = "name"
    RANGE = "range"
    START = "start"
    STOP = "stop"
    STEP = "step"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    DTYPE = "dtype"
    TZ = "tz"
    FREQ = "freq"
    VALUES_I8 = "values_i8"
    VALUES = "values"
    LEVELS = "levels"
    CODES = "codes"
    NAMES = "names"
    SORTORDER = "sortorder"
    NLEVELS = "nlevels"


class AxisType(StrEnum):
    MULTIINDEX = "multiindex"


STRING_MISSING_VALUE_SENTINEL = "<MISSING_STRING_VALUE_SENTINEL_df_npy>"
NUMPY_DTYPE_FLOAT64 = "float64"
NUMPY_DTYPE_UNICODE = "U"
PANDAS_NULLABLE_INT_DTYPE = "Int64"
PANDAS_BOOL_DTYPE = "bool"
PANDAS_NULLABLE_BOOL_DTYPE = "boolean"
DEFAULT_TIME_UNIT = "ns"
NPY_SUFFIX = ".npy"
JSON_SUFFIX = ".json"
ARRAY_ORDER_FORTRAN = "F"


STRING_DTYPE_NAMES = {"object", "string", "str"}
