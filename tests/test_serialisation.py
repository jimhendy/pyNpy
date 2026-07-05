import pandas as pd
import pytest

from df_npy import NpySerializer, get_columns

TEST_DF_WIDTH = 2
TEST_DF_LENGTH = 3

PARAMETERISED_TYPE_DFS = [
    ("int", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})),
    ("float", pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [4.4, 5.5, 6.6]})),
    ("str", pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})),
    ("int_with_nan", pd.DataFrame({"a": [1, 2, None], "b": [4, None, 6]})),
    (
        "dates",
        pd.DataFrame(
            {
                "a": pd.date_range("2023-01-01", periods=3),
                "b": pd.date_range("2023-02-01", periods=3),
            }
        ),
    ),
]

UNSUPPORTED_TYPE_DFS = [
    ("mixed", pd.DataFrame({"a": [1, 2.2, "three"], "b": [4.4, "five", 6]})),
]

PARAMETERISED_INDICES = [
    ("default", pd.RangeIndex(start=0, stop=3, step=1)),
    ("int", pd.Index([10, 20, 30])),
    ("str", pd.Index(["a", "b", "c"])),
    ("datetime", pd.date_range("2023-01-01", periods=3)),
    ("datetime_with_tz_utc", pd.date_range("2023-01-01", periods=3, tz="UTC")),
    (
        "datetime_with_tz_london",
        pd.date_range("2023-01-01", periods=3, tz="Europe/London"),
    ),
    ("range", pd.RangeIndex(start=5, stop=20, step=5)),
    (
        "multiindex",
        pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1)], names=["letter", "number"]
        ),
    ),
]


class TestSimpleTypes:
    @pytest.mark.parametrize("type_name, df", PARAMETERISED_TYPE_DFS)
    def test_serialisation_deserialisation(self, type_name, df, tmp_path):
        file_path = tmp_path / "test.npy"
        NpySerializer.to_npy(df, file_path)
        assert file_path.exists()
        deserialized = NpySerializer.from_npy(file_path)
        pd.testing.assert_frame_equal(df, deserialized)

    @pytest.mark.parametrize("type_name, df", PARAMETERISED_TYPE_DFS)
    @pytest.mark.parametrize("index_name, index", PARAMETERISED_INDICES)
    def test_serialisation_deserialisation_with_index(
        self, index_name, index, type_name, df, tmp_path
    ):
        df = df.copy()
        df.index = index
        file_path = tmp_path / "test.npy"
        NpySerializer.to_npy(df, file_path)
        assert file_path.exists()
        deserialized = NpySerializer.from_npy(file_path)
        pd.testing.assert_frame_equal(df, deserialized)


class TestUnsupportedTypes:
    @pytest.mark.parametrize("type_name, df", UNSUPPORTED_TYPE_DFS)
    def test_serialisation_rejects_pickle_backed_frames(self, type_name, df, tmp_path):
        file_path = tmp_path / "test.npy"

        with pytest.raises(
            ValueError, match="Pickle-backed object serialization is disabled"
        ):
            NpySerializer.to_npy(df, file_path)


def test_get_columns_top_level_returns_string_set(tmp_path):
    df = pd.DataFrame({"a": [1, 2], 2: [3, 4], "c": [5, 6]})
    file_path = tmp_path / "test.npy"

    NpySerializer.to_npy(df, file_path)

    assert get_columns(file_path) == {"a", "2", "c"}


def test_serializer_get_columns_raises_on_missing_columns_metadata(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.npy"
    metadata_path = file_path.with_suffix(".json")

    NpySerializer.to_npy(df, file_path)
    metadata = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(
        metadata.replace('"columns":', '"_columns":'), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Metadata is missing required key: columns"):
        NpySerializer.get_columns(file_path)


def test_serializer_multiindex_multicolumn(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [("A", 1), ("A", 2), ("B", 1)], names=["letter", "number"]
    )
    columns = pd.MultiIndex.from_tuples(
        [("X", "x"), ("X", "y"), ("Y", "z")], names=["group", "subgroup"]
    )
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=columns)

    file_path = tmp_path / "test.npy"
    NpySerializer.to_npy(df, file_path)
    deserialized = NpySerializer.from_npy(file_path)

    pd.testing.assert_frame_equal(df, deserialized)


def test_serializer_multiindex_multicolumn_with_dates(tmp_path):
    dates = pd.date_range("2023-01-01", periods=3)
    index = pd.MultiIndex.from_tuples(
        [(date, i) for i, date in enumerate(dates)],
        names=["date", "number"],
    )

    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], index=index, columns=index
    )

    file_path = tmp_path / "test.npy"
    NpySerializer.to_npy(df, file_path)
    deserialized = NpySerializer.from_npy(file_path)

    pd.testing.assert_frame_equal(df, deserialized)
