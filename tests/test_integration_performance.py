from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from df_npy import NpySerializer

pytestmark = [pytest.mark.integration, pytest.mark.performance]
MAX_SUBSET_SLOWDOWN_FACTOR = 2.5


def _make_large_dataframe(rows: int = 80_000, cols: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(rows, cols)).astype(np.float64)
    columns = pd.Index([f"col_{i:03d}" for i in range(cols)])
    return pd.DataFrame(data, columns=columns)


def _median_runtime(fn, repeats: int = 5) -> float:
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        timings.append(perf_counter() - start)
    return statistics.median(timings)


def test_identifier_subset_load_is_not_much_slower_than_full_load_for_large_file(
    tmp_path: Path,
) -> None:
    df = _make_large_dataframe()
    subset_cols = list(df.columns[:6])
    file_path = tmp_path / "large.npy"

    NpySerializer.to_npy(df, file_path)

    expected_subset = df[subset_cols]
    loaded_subset = NpySerializer.from_npy(file_path, identifiers=subset_cols)
    pd.testing.assert_frame_equal(loaded_subset, expected_subset)

    subset_time = _median_runtime(
        lambda: NpySerializer.from_npy(file_path, identifiers=subset_cols),
        repeats=5,
    )
    full_time = _median_runtime(lambda: NpySerializer.from_npy(file_path), repeats=3)

    assert subset_time <= full_time * MAX_SUBSET_SLOWDOWN_FACTOR, (
        "Subset loading should remain in the same performance range as full loading "
        "for large arrays. "
        f"subset={subset_time:.4f}s, full={full_time:.4f}s"
    )


def test_threadpool_subset_load_is_not_much_slower_than_threadpool_full_load(
    tmp_path: Path,
) -> None:
    df = _make_large_dataframe(rows=60_000, cols=64)
    subset_cols = list(df.columns[:6])

    paths: list[Path] = []
    for i in range(4):
        path = tmp_path / f"large_{i}.npy"
        NpySerializer.to_npy(df, path)
        paths.append(path)

    def load_full(path: Path) -> pd.DataFrame:
        return NpySerializer.from_npy(path)

    def load_subset(path: Path) -> pd.DataFrame:
        return NpySerializer.from_npy(path, identifiers=subset_cols)

    def run_threadpool_subset() -> list[pd.DataFrame]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(load_subset, paths))

    def run_threadpool_full() -> list[pd.DataFrame]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(load_full, paths))

    subset_results = run_threadpool_subset()
    full_results = run_threadpool_full()

    subset_time = _median_runtime(run_threadpool_subset, repeats=3)
    full_time = _median_runtime(run_threadpool_full, repeats=3)

    for subset_df in subset_results:
        pd.testing.assert_frame_equal(subset_df, df[subset_cols])
    for full_df in full_results:
        pd.testing.assert_frame_equal(full_df, df)

    assert subset_time <= full_time * MAX_SUBSET_SLOWDOWN_FACTOR, (
        "Threadpool subset loading should stay in the same performance range as "
        "threadpool full loading. "
        f"subset={subset_time:.4f}s, full={full_time:.4f}s"
    )


def test_subset_load_timing_comparison_with_parquet_and_feather(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    df = _make_large_dataframe(rows=50_000, cols=48)
    subset_cols = list(df.columns[:6])

    npy_path = tmp_path / "large.npy"
    parquet_path = tmp_path / "large.parquet"
    feather_path = tmp_path / "large.feather"

    NpySerializer.to_npy(df, npy_path)
    df.to_parquet(parquet_path, index=False)
    df.to_feather(feather_path)

    npy_subset = NpySerializer.from_npy(npy_path, identifiers=subset_cols)
    parquet_subset = pd.read_parquet(parquet_path, columns=subset_cols)
    feather_subset = pd.read_feather(feather_path, columns=subset_cols)

    pd.testing.assert_frame_equal(npy_subset.reset_index(drop=True), parquet_subset)
    pd.testing.assert_frame_equal(npy_subset.reset_index(drop=True), feather_subset)

    npy_time = _median_runtime(
        lambda: NpySerializer.from_npy(npy_path, identifiers=subset_cols),
        repeats=5,
    )
    parquet_time = _median_runtime(
        lambda: pd.read_parquet(parquet_path, columns=subset_cols),
        repeats=5,
    )
    feather_time = _median_runtime(
        lambda: pd.read_feather(feather_path, columns=subset_cols),
        repeats=5,
    )

    slowest_other = max(parquet_time, feather_time)
    assert npy_time <= slowest_other * 3.0, (
        "Npy subset load should be in a comparable range to parquet/feather subset loads. "
        f"npy={npy_time:.4f}s, parquet={parquet_time:.4f}s, feather={feather_time:.4f}s"
    )
