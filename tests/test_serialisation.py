import pandas as pd
import pytest

from df_npy import NpySerializer

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
