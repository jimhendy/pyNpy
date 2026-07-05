"""Benchmark df-npy against common pandas serialization formats."""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from loguru import logger

from df_npy import NpySerializer

plt: Any = None
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    pass


README_START = "<!-- BENCHMARK_RESULTS_START -->"
README_END = "<!-- BENCHMARK_RESULTS_END -->"
SCALING_FORMATS = ("df-npy", "pickle", "parquet", "feather")
SCALING_RATIO_FORMATS = ("pickle", "parquet", "feather")
MULTITHREAD_FORMATS = ("df-npy", "pickle", "parquet", "feather")


@dataclass(frozen=True)
class BenchmarkResult:
    """Store benchmark timing data for one serialization format."""

    name: str
    write_seconds: float | None
    read_seconds: float | None
    error: str | None = None


@dataclass(frozen=True)
class MultiThreadResult:
    """Store concurrent subset-read benchmark data for one size+format."""

    target_mb: float
    actual_bytes: int
    files: int
    workers: int
    format_name: str
    read_seconds: float | None
    error: str | None = None
    samples: tuple[float, ...] = ()


def _drop_file_cache(path: Path) -> str:
    """Best-effort advisory drop of OS page cache for a file path."""
    if not path.exists():
        return "skipped (file missing)"

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return "unsupported on this platform"

    try:
        os.sync()
        with path.open("rb") as handle:
            os.posix_fadvise(
                handle.fileno(),
                0,
                0,
                os.POSIX_FADV_DONTNEED,
            )
        return "posix_fadvise(POSIX_FADV_DONTNEED)"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"failed: {exc}"


def _cache_policy_description() -> str:
    """Describe the cache eviction mechanism available on this host."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return "unsupported on this platform"
    return "posix_fadvise(POSIX_FADV_DONTNEED)"


def _make_dataframe(rows: int, cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(rows, cols)).astype(np.float64)
    data[::97, ::17] = np.nan
    columns = [f"col_{i:03d}" for i in range(cols)]
    return pd.DataFrame(data, columns=columns)


def _median_seconds(
    fn: Callable[[], object],
    repeats: int,
    before_each: Callable[[], object] | None = None,
) -> float:
    return statistics.median(_timing_samples(fn, repeats, before_each))


def _timing_samples(
    fn: Callable[[], object],
    repeats: int,
    before_each: Callable[[], object] | None = None,
) -> list[float]:
    """Return all timing samples so callers can compute uncertainty bands."""
    timings: list[float] = []
    for _ in range(repeats):
        if before_each is not None:
            before_each()
        start = perf_counter()
        fn()
        timings.append(perf_counter() - start)
    return timings


def _select_subset_columns(cols: int, subset_fraction: float, seed: int) -> list[str]:
    """Select a reproducible random subset of columns."""
    subset_count = min(cols, max(1, int(round(cols * subset_fraction))))
    rng = np.random.default_rng(seed)
    subset_indices = sorted(rng.choice(cols, size=subset_count, replace=False).tolist())
    return [f"col_{index:03d}" for index in subset_indices]


def _bench_format(
    name: str,
    write_fn: Callable[[Path, pd.DataFrame], object],
    read_fn: Callable[[Path], object],
    path: Path,
    df: pd.DataFrame,
    repeats: int,
) -> BenchmarkResult:
    try:
        write_fn(path, df)
        _drop_file_cache(path)
        loaded = cast(pd.DataFrame, read_fn(path))
        if loaded.shape != df.shape:
            raise ValueError(
                f"Round-trip shape mismatch for {name}: {loaded.shape} != {df.shape}"
            )

        write_seconds = _median_seconds(
            lambda: write_fn(path, df),
            repeats,
            before_each=lambda: _drop_file_cache(path),
        )
        read_seconds = _median_seconds(
            lambda: read_fn(path),
            repeats,
            before_each=lambda: _drop_file_cache(path),
        )
        return BenchmarkResult(
            name=name,
            write_seconds=write_seconds,
            read_seconds=read_seconds,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return BenchmarkResult(
            name=name,
            write_seconds=None,
            read_seconds=None,
            error=str(exc),
        )


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"


def _relative(value: float | None, baseline: float | None) -> str:
    if value is None or baseline in (None, 0.0):
        return "n/a"
    return f"{value / baseline:.2f}x"


def _build_markdown_table(results: list[BenchmarkResult]) -> str:
    baseline = next((item for item in results if item.name == "df-npy"), None)
    baseline_write = baseline.write_seconds if baseline else None
    baseline_read = baseline.read_seconds if baseline else None

    lines = [
        "| Format | Write (median) | Read (median) | "
        "Write vs df-npy | Read vs df-npy | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        notes = result.error or ""
        lines.append(
            "| "
            f"{result.name} | {_format_seconds(result.write_seconds)} | "
            f"{_format_seconds(result.read_seconds)} "
            f"| {_relative(result.write_seconds, baseline_write)} | "
            f"{_relative(result.read_seconds, baseline_read)} "
            f"| {notes} |"
        )
    return "\n".join(lines)


def _plot_results(
    results: list[BenchmarkResult],
    output_path: Path,
    rows: int,
    cols: int,
) -> bool:
    if plt is None:
        return False

    usable = [
        item
        for item in results
        if item.write_seconds is not None and item.read_seconds is not None
    ]
    if not usable:
        return False

    labels = [item.name for item in usable]
    write_vals = np.array([item.write_seconds for item in usable], dtype=float)
    read_vals = np.array([item.read_seconds for item in usable], dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, write_vals, width, label="write")
    ax.bar(x + width / 2, read_vals, width, label="read")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds (log scale)")
    ax.set_title(f"DataFrame serialisation benchmark ({rows} x {cols})")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _bench_scaling(
    cols: int,
    max_bytes: int,
    num_points: int,
    repeats: int,
    seed: int,
    subset_fraction: float,
) -> list[dict[str, object]]:
    """Benchmark read scaling for df-npy, pickle, parquet, and feather.

    Returns entries with shape:
    {
        "bytes": int,
        "full": {format_name: seconds | None},
        "subset": {format_name: seconds | None},
    }
    """
    bytes_per_row = cols * 8  # float64
    min_rows = 10
    max_rows = max(min_rows + 1, max_bytes // bytes_per_row)
    subset_columns = _select_subset_columns(cols, subset_fraction, seed + 1_001)

    row_counts = np.unique(
        np.round(
            np.logspace(np.log10(min_rows), np.log10(max_rows), num=num_points)
        ).astype(int)
    )

    scaling_results: list[dict[str, object]] = []
    for rows in row_counts:
        df = _make_dataframe(int(rows), cols, seed)
        df_bytes = int(rows) * cols * 8

        with TemporaryDirectory() as tmp:
            npy_path = Path(tmp) / "data.npy"
            pkl_path = Path(tmp) / "data.pkl"
            parquet_path = Path(tmp) / "data.parquet"
            feather_path = Path(tmp) / "data.feather"

            full_times: dict[str, float | None] = {
                name: None for name in SCALING_FORMATS
            }
            subset_times: dict[str, float | None] = {
                name: None for name in SCALING_FORMATS
            }
            full_samples: dict[str, list[float]] = {
                name: [] for name in SCALING_FORMATS
            }
            subset_samples: dict[str, list[float]] = {
                name: [] for name in SCALING_FORMATS
            }

            try:
                NpySerializer.to_npy(df, npy_path)

                # np.load uses mmap_mode="r" by default, so the array is memory-mapped
                # and no data is read until each page is first accessed.  Force full
                # materialisation here so we measure actual I/O, not just mmap setup.
                def _npy_read_all_and_materialise() -> None:
                    result = NpySerializer.from_npy(npy_path)
                    _ = result.to_numpy().sum()  # touch every page

                def _npy_read_subset_and_materialise() -> None:
                    result = NpySerializer.from_npy(
                        npy_path,
                        identifiers=subset_columns,
                    )
                    _ = result.to_numpy().sum()  # touch every page in subset

                full_samples["df-npy"] = _timing_samples(
                    _npy_read_all_and_materialise,
                    repeats,
                    before_each=lambda: _drop_file_cache(npy_path),
                )
                full_times["df-npy"] = statistics.median(full_samples["df-npy"])
                subset_samples["df-npy"] = _timing_samples(
                    _npy_read_subset_and_materialise,
                    repeats,
                    before_each=lambda: _drop_file_cache(npy_path),
                )
                subset_times["df-npy"] = statistics.median(subset_samples["df-npy"])
            except Exception as exc:
                logger.warning(f"df-npy failed at rows={rows}: {exc}")

            try:
                with open(pkl_path, "wb") as fh:
                    pickle.dump(df, fh)

                full_samples["pickle"] = _timing_samples(
                    lambda: pd.read_pickle(pkl_path),
                    repeats,
                    before_each=lambda: _drop_file_cache(pkl_path),
                )
                full_times["pickle"] = statistics.median(full_samples["pickle"])
                subset_samples["pickle"] = _timing_samples(
                    lambda: pd.read_pickle(pkl_path)[subset_columns],
                    repeats,
                    before_each=lambda: _drop_file_cache(pkl_path),
                )
                subset_times["pickle"] = statistics.median(subset_samples["pickle"])
            except Exception as exc:
                logger.warning(f"pickle failed at rows={rows}: {exc}")

            try:
                df.to_parquet(parquet_path, index=False)

                full_samples["parquet"] = _timing_samples(
                    lambda: pd.read_parquet(parquet_path),
                    repeats,
                    before_each=lambda: _drop_file_cache(parquet_path),
                )
                full_times["parquet"] = statistics.median(full_samples["parquet"])
                subset_samples["parquet"] = _timing_samples(
                    lambda: pd.read_parquet(parquet_path, columns=subset_columns),
                    repeats,
                    before_each=lambda: _drop_file_cache(parquet_path),
                )
                subset_times["parquet"] = statistics.median(subset_samples["parquet"])
            except Exception as exc:
                logger.warning(f"parquet failed at rows={rows}: {exc}")

            try:
                df.to_feather(feather_path)

                full_samples["feather"] = _timing_samples(
                    lambda: pd.read_feather(feather_path),
                    repeats,
                    before_each=lambda: _drop_file_cache(feather_path),
                )
                full_times["feather"] = statistics.median(full_samples["feather"])
                subset_samples["feather"] = _timing_samples(
                    lambda: pd.read_feather(feather_path, columns=subset_columns),
                    repeats,
                    before_each=lambda: _drop_file_cache(feather_path),
                )
                subset_times["feather"] = statistics.median(subset_samples["feather"])
            except Exception as exc:
                logger.warning(f"feather failed at rows={rows}: {exc}")

        scaling_results.append(
            {
                "bytes": df_bytes,
                "full": full_times,
                "subset": subset_times,
                "full_samples": full_samples,
                "subset_samples": subset_samples,
            }
        )
        logger.debug(
            f"scaling rows={rows:,} bytes={df_bytes:,} "
            f"full={full_times} subset={subset_times}"
        )

    return scaling_results


def _plot_scaling(
    scaling_data: list[dict[str, object]],
    output_path: Path,
    mode: str,
) -> bool:
    """Two-subplot line plot: read time vs size (top) and ratios vs df-npy (bottom)."""
    if plt is None:
        return False

    if mode not in {"full", "subset"}:
        raise ValueError(f"Unknown scaling mode: {mode}")

    sizes = np.array([entry["bytes"] for entry in scaling_data], dtype=float)
    if sizes.size < 2:
        return False

    sample_key = "full_samples" if mode == "full" else "subset_samples"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [7, 3]},
        sharex=True,
    )

    for format_name in SCALING_FORMATS:
        point_sizes: list[float] = []
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []

        for entry in scaling_data:
            samples = cast(dict[str, list[float]], entry[sample_key]).get(format_name)
            if not samples:
                continue
            sample_arr = np.array(samples, dtype=float)
            point_sizes.append(float(cast(int, entry["bytes"])))
            medians.append(float(np.median(sample_arr)))
            lows.append(float(np.percentile(sample_arr, 25)))
            highs.append(float(np.percentile(sample_arr, 75)))

        if len(point_sizes) < 2:
            continue

        x_vals = np.array(point_sizes, dtype=float)
        y_vals = np.array(medians, dtype=float)
        low_vals = np.array(lows, dtype=float)
        high_vals = np.array(highs, dtype=float)

        ax_top.plot(
            x_vals,
            y_vals,
            marker="o",
            markersize=4,
            label=format_name,
        )
        ax_top.fill_between(x_vals, low_vals, high_vals, alpha=0.14)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Read + materialise time (s, log scale)")
    if mode == "full":
        ax_top.set_title(
            "Full DataFrame read time vs size: df-npy, pickle, parquet, feather"
        )
    else:
        ax_top.set_title(
            "Subset read time vs size (50% columns): df-npy, pickle, parquet, feather"
        )
    ax_top.legend()
    ax_top.grid(True, which="both", alpha=0.35, linestyle="--")
    ax_top.set_axisbelow(True)

    for format_name in SCALING_RATIO_FORMATS:
        ratio_sizes: list[float] = []
        ratio_medians: list[float] = []
        ratio_lows: list[float] = []
        ratio_highs: list[float] = []

        for entry in scaling_data:
            base_samples = cast(dict[str, list[float]], entry[sample_key]).get("df-npy")
            fmt_samples = cast(dict[str, list[float]], entry[sample_key]).get(
                format_name
            )
            if not base_samples or not fmt_samples:
                continue
            count = min(len(base_samples), len(fmt_samples))
            ratio_sample_list = [
                fmt_samples[index] / base_samples[index]
                for index in range(count)
                if base_samples[index] > 0
            ]
            if not ratio_sample_list:
                continue
            ratio_arr = np.array(ratio_sample_list, dtype=float)
            ratio_sizes.append(float(cast(int, entry["bytes"])))
            ratio_medians.append(float(np.median(ratio_arr)))
            ratio_lows.append(float(np.percentile(ratio_arr, 25)))
            ratio_highs.append(float(np.percentile(ratio_arr, 75)))

        if len(ratio_sizes) < 2:
            continue

        ratio_size_arr = np.array(ratio_sizes, dtype=float)
        ratio_values = np.array(ratio_medians, dtype=float)
        ratio_low_arr = np.array(ratio_lows, dtype=float)
        ratio_high_arr = np.array(ratio_highs, dtype=float)
        ax_bot.plot(
            ratio_size_arr,
            ratio_values,
            marker="o",
            markersize=4,
            label=f"{format_name} / df-npy",
        )
        ax_bot.fill_between(
            ratio_size_arr,
            ratio_low_arr,
            ratio_high_arr,
            alpha=0.14,
        )

    ax_bot.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        label="ratio = 1",
    )
    ax_bot.set_xscale("log")
    ax_bot.set_ylabel("Ratio (format / df-npy)")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, which="both", alpha=0.35, linestyle="--")
    ax_bot.set_axisbelow(True)

    def _fmt_bytes(x: float, _: object) -> str:
        if x >= 1e9:
            return f"{x / 1e9:.1f} GB"
        if x >= 1e6:
            return f"{x / 1e6:.0f} MB"
        return f"{x / 1e3:.0f} KB"

    from matplotlib.ticker import FuncFormatter

    ax_bot.xaxis.set_major_formatter(FuncFormatter(_fmt_bytes))
    ax_bot.set_xlabel("DataFrame size (uncompressed float64)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _parse_target_mbs(raw: str) -> list[float]:
    """Parse comma-separated MB sizes like '1,100'."""
    values: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError("Target MB values must be > 0")
        values.append(value)
    if not values:
        raise ValueError("At least one target MB value is required")
    return sorted(set(values))


def _expand_target_mbs(target_mbs: list[float], scan_points: int) -> list[float]:
    """Expand min/max MB bounds to a log-spaced scan, preserving explicit points."""
    base = sorted(set(target_mbs))
    if len(base) < 2 or scan_points <= len(base):
        return base

    lo = min(base)
    hi = max(base)
    if lo == hi:
        return base

    generated = np.logspace(np.log10(lo), np.log10(hi), num=scan_points)
    merged = sorted({*base, *(float(x) for x in generated)})
    return [round(x, 6) for x in merged]


def _rows_for_target_mb(cols: int, target_mb: float) -> int:
    """Approximate row count for a float64 DataFrame near target MB."""
    target_bytes = int(target_mb * 1024 * 1024)
    bytes_per_row = cols * 8
    return max(10, target_bytes // max(1, bytes_per_row))


def _build_multithread_markdown(results: list[MultiThreadResult]) -> str:
    """Build markdown table for concurrent subset-read benchmark results."""
    lines = [
        "| Target size | Actual size | Format | Files | Workers | "
        "Time (median) | p25 | p75 | Files/s | Notes |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(results, key=lambda x: (x.target_mb, x.format_name)):
        files_per_second = (
            item.files / item.read_seconds
            if item.read_seconds is not None and item.read_seconds > 0
            else None
        )
        notes = item.error or ""
        sample_arr = np.array(item.samples, dtype=float)
        p25 = float(np.percentile(sample_arr, 25)) if sample_arr.size else None
        p75 = float(np.percentile(sample_arr, 75)) if sample_arr.size else None
        files_per_second_str = (
            f"{files_per_second:.2f}" if files_per_second is not None else "n/a"
        )
        lines.append(
            "| "
            f"{item.target_mb:.1f} MB | "
            f"{item.actual_bytes / (1024 * 1024):.1f} MB | "
            f"{item.format_name} | "
            f"{item.files} | "
            f"{item.workers} | "
            f"{_format_seconds(item.read_seconds)} | "
            f"{_format_seconds(p25)} | "
            f"{_format_seconds(p75)} | "
            f"{files_per_second_str} | "
            f"{notes} |"
        )
    return "\n".join(lines)


def _plot_multithread_results(
    results: list[MultiThreadResult],
    output_path: Path,
) -> bool:
    """Plot concurrent read wall-time and ratio vs df-npy with error bands."""
    if plt is None:
        return False

    sizes = sorted({item.target_mb for item in results})
    if len(sizes) < 2:
        return False

    indexed = {(item.target_mb, item.format_name): item for item in results}

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [7, 3]},
        sharex=True,
    )

    for format_name in MULTITHREAD_FORMATS:
        x_vals: list[float] = []
        med_vals: list[float] = []
        low_vals: list[float] = []
        high_vals: list[float] = []

        for size in sizes:
            item = indexed.get((size, format_name))
            if item is None or item.read_seconds is None:
                continue
            samples = np.array(item.samples, dtype=float)
            if samples.size == 0:
                continue
            x_vals.append(size)
            med_vals.append(float(np.median(samples)))
            low_vals.append(float(np.percentile(samples, 25)))
            high_vals.append(float(np.percentile(samples, 75)))

        if len(x_vals) < 2:
            continue

        x_arr = np.array(x_vals, dtype=float)
        med_arr = np.array(med_vals, dtype=float)
        low_arr = np.array(low_vals, dtype=float)
        high_arr = np.array(high_vals, dtype=float)

        ax_top.plot(x_arr, med_arr, marker="o", markersize=4, label=format_name)
        ax_top.fill_between(x_arr, low_arr, high_arr, alpha=0.14)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Concurrent read wall time (s, log)")
    ax_top.set_title("Multithreaded subset read benchmark (50% random columns)")
    ax_top.grid(True, which="both", alpha=0.35, linestyle="--")
    ax_top.set_axisbelow(True)
    ax_top.legend()

    for format_name in SCALING_RATIO_FORMATS:
        ratio_x: list[float] = []
        ratio_med: list[float] = []
        ratio_low: list[float] = []
        ratio_high: list[float] = []

        for size in sizes:
            base = indexed.get((size, "df-npy"))
            other = indexed.get((size, format_name))
            if base is None or other is None:
                continue
            if not base.samples or not other.samples:
                continue
            count = min(len(base.samples), len(other.samples))
            ratio_samples = [
                other.samples[i] / base.samples[i]
                for i in range(count)
                if base.samples[i] > 0
            ]
            if not ratio_samples:
                continue

            ratio_arr = np.array(ratio_samples, dtype=float)
            ratio_x.append(size)
            ratio_med.append(float(np.median(ratio_arr)))
            ratio_low.append(float(np.percentile(ratio_arr, 25)))
            ratio_high.append(float(np.percentile(ratio_arr, 75)))

        if len(ratio_x) < 2:
            continue

        ratio_x_arr = np.array(ratio_x, dtype=float)
        ratio_med_arr = np.array(ratio_med, dtype=float)
        ratio_low_arr = np.array(ratio_low, dtype=float)
        ratio_high_arr = np.array(ratio_high, dtype=float)
        ax_bot.plot(
            ratio_x_arr,
            ratio_med_arr,
            marker="o",
            markersize=4,
            label=f"{format_name} / df-npy",
        )
        ax_bot.fill_between(ratio_x_arr, ratio_low_arr, ratio_high_arr, alpha=0.14)

    ax_bot.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="ratio = 1")
    ax_bot.set_xscale("log")
    ax_bot.set_ylabel("Ratio (format / df-npy)")
    ax_bot.set_xlabel("DataFrame size (MB, uncompressed float64)")
    ax_bot.grid(True, which="both", alpha=0.35, linestyle="--")
    ax_bot.set_axisbelow(True)
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _bench_multithread_subset_reads(
    target_mbs: list[float],
    cols: int,
    seed: int,
    subset_fraction: float,
    repeats: int,
    files_per_size: int,
    workers: int,
) -> list[MultiThreadResult]:
    """Benchmark concurrent subset reads across formats for multiple target sizes."""
    results: list[MultiThreadResult] = []

    subset_columns = _select_subset_columns(cols, subset_fraction, seed + 8_888)

    for size_index, target_mb in enumerate(target_mbs):
        rows = _rows_for_target_mb(cols, target_mb)
        dataset_seed = seed + size_index
        df = _make_dataframe(rows, cols, dataset_seed)
        actual_bytes = int(rows) * cols * 8

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            per_format_paths: dict[str, list[Path]] = {
                fmt: [] for fmt in MULTITHREAD_FORMATS
            }

            for file_index in range(files_per_size):
                npy_path = tmp_path / f"npy_{file_index:03d}.npy"
                pkl_path = tmp_path / f"pickle_{file_index:03d}.pkl"
                parquet_path = tmp_path / f"parquet_{file_index:03d}.parquet"
                feather_path = tmp_path / f"feather_{file_index:03d}.feather"

                NpySerializer.to_npy(df, npy_path)
                df.to_pickle(pkl_path)
                df.to_parquet(parquet_path, index=False)
                df.to_feather(feather_path)

                per_format_paths["df-npy"].append(npy_path)
                per_format_paths["pickle"].append(pkl_path)
                per_format_paths["parquet"].append(parquet_path)
                per_format_paths["feather"].append(feather_path)

            # Flush all written data to storage and advise the OS to evict the
            # page-cache entries for every file.  We do this once after writing
            # ALL files (not per-format) so that the cache state is as cold as
            # posix_fadvise can make it before any timed read begins.
            # Note: posix_fadvise is advisory; the kernel may honour it only
            # partially for recently-written dirty pages.
            os.sync()
            for fmt_paths in per_format_paths.values():
                for p in fmt_paths:
                    try:
                        with p.open("rb") as fh:
                            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    except Exception:
                        pass

            format_readers: dict[str, Callable[[Path], float]] = {
                "df-npy": lambda path: float(
                    NpySerializer.from_npy(path, identifiers=subset_columns)
                    .to_numpy(copy=False)
                    .sum()
                ),
                "pickle": lambda path: float(
                    pd.read_pickle(path)[subset_columns].to_numpy(copy=False).sum()
                ),
                "parquet": lambda path: float(
                    pd.read_parquet(path, columns=subset_columns)
                    .to_numpy(copy=False)
                    .sum()
                ),
                "feather": lambda path: float(
                    pd.read_feather(path, columns=subset_columns)
                    .to_numpy(copy=False)
                    .sum()
                ),
            }

            for format_name in MULTITHREAD_FORMATS:
                paths = per_format_paths[format_name]
                reader = format_readers[format_name]

                try:

                    def run_once() -> None:
                        for p in paths:
                            _drop_file_cache(p)
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            _ = list(executor.map(reader, paths))

                    samples = _timing_samples(run_once, repeats)
                    seconds = statistics.median(samples)
                    error = None
                except Exception as exc:
                    seconds = None
                    error = str(exc)
                    samples = []

                results.append(
                    MultiThreadResult(
                        target_mb=target_mb,
                        actual_bytes=actual_bytes,
                        files=files_per_size,
                        workers=workers,
                        format_name=format_name,
                        read_seconds=seconds,
                        error=error,
                        samples=tuple(samples),
                    )
                )

    return results


def _update_readme(readme_path: Path, section: str) -> None:
    content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    block = f"{README_START}\n{section}\n{README_END}"

    if README_START in content and README_END in content:
        start = content.index(README_START)
        end = content.index(README_END) + len(README_END)
        updated = content[:start] + block + content[end:]
    else:
        spacer = "\n\n" if content and not content.endswith("\n") else "\n"
        updated = content + spacer + "## Benchmark\n\n" + block + "\n"

    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Run benchmark and write markdown/image outputs."""
    parser = argparse.ArgumentParser(
        description="Benchmark df-npy against parquet, feather, hdf5, and pickle.",
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats per format; higher reduces variance but takes longer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--no-readme-update", action="store_true")
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Also generate the read-time-vs-size scaling line plot.",
    )
    parser.add_argument(
        "--scaling-max-gb",
        type=float,
        default=5.0,
        metavar="GB",
        help="Upper bound for the scaling plot in gigabytes (default: 5.0).",
    )
    parser.add_argument(
        "--scaling-points",
        type=int,
        default=20,
        metavar="N",
        help="Number of log-spaced size points for the scaling plot (default: 20).",
    )
    parser.add_argument(
        "--scaling-repeats",
        type=int,
        default=3,
        metavar="N",
        help="Repeats per size point in the scaling plot (default: 3).",
    )
    parser.add_argument(
        "--scaling-subset-fraction",
        type=float,
        default=0.5,
        metavar="F",
        help="Fraction of columns used for subset-read scaling plot (default: 0.5).",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Run multithreaded subset-read benchmark across formats.",
    )
    parser.add_argument(
        "--multithread-target-mb",
        type=str,
        default="1,100",
        metavar="CSV",
        help="Comma-separated target DataFrame sizes in MB (default: 1,100).",
    )
    parser.add_argument(
        "--multithread-scan-points",
        type=int,
        default=10,
        metavar="N",
        help="Log-spaced points between min/max target MB (default: 10).",
    )
    parser.add_argument(
        "--multithread-files",
        type=int,
        default=24,
        metavar="N",
        help="Number of files to read concurrently per target size (default: 24).",
    )
    parser.add_argument(
        "--multithread-workers",
        type=int,
        default=8,
        metavar="N",
        help="Thread pool worker count for concurrent reads (default: 8).",
    )
    parser.add_argument(
        "--multithread-repeats",
        type=int,
        default=7,
        metavar="N",
        help="Median repeats for multithread benchmark (default: 7).",
    )
    args = parser.parse_args()

    # Keep benchmark output concise by silencing serializer internals.
    logger.disable("df_npy")

    df = _make_dataframe(args.rows, args.cols, args.seed)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cache_policy = _cache_policy_description()
        candidates: list[
            tuple[
                str,
                Path,
                Callable[[Path, pd.DataFrame], object],
                Callable[[Path], object],
            ]
        ] = [
            (
                "df-npy",
                tmp_path / "data.npy",
                lambda p, data: NpySerializer.to_npy(data, p),
                lambda p: NpySerializer.from_npy(p),
            ),
            (
                "parquet",
                tmp_path / "data.parquet",
                lambda p, data: data.to_parquet(p, index=False),
                lambda p: pd.read_parquet(p),
            ),
            (
                "feather",
                tmp_path / "data.feather",
                lambda p, data: data.to_feather(p),
                lambda p: pd.read_feather(p),
            ),
            (
                "hdf5",
                tmp_path / "data.h5",
                lambda p, data: data.to_hdf(p, key="data", mode="w"),
                lambda p: pd.read_hdf(p, key="data"),
            ),
            (
                "pickle",
                tmp_path / "data.pkl",
                lambda p, data: data.to_pickle(p),
                lambda p: pd.read_pickle(p),
            ),
        ]

        results = [
            _bench_format(name, write_fn, read_fn, path, df, args.repeats)
            for name, path, write_fn, read_fn in candidates
        ]

    chart_path = args.output_dir / "serialisation_benchmark.png"
    has_chart = _plot_results(results, chart_path, args.rows, args.cols)

    table = _build_markdown_table(results)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    notes = [
        "Generated by scripts/benchmark_serialisation.py.",
        "Dataset shape: "
        f"{args.rows} x {args.cols}, repeats={args.repeats}, seed={args.seed}.",
        "Cache policy between timed write/read ops: "
        f"{cache_policy} (best-effort advisory; does not guarantee cold cache).",
        "Note: Write times have high variance due to filesystem I/O variability. "
        "Treat as approximate/directional, not precise benchmarks.",
        f"Generated at: {generated_at}.",
    ]

    md_lines = [
        "### Serialisation benchmark",
        "",
        *[f"- {line}" for line in notes],
        "",
        table,
    ]

    if has_chart:
        md_lines.extend(
            ["", "![Serialisation benchmark](benchmark/serialisation_benchmark.png)"]
        )
    else:
        md_lines.extend(
            [
                "",
                "Chart not generated (matplotlib not installed).",
            ]
        )

    section = "\n".join(md_lines)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "serialisation_benchmark.md").write_text(
        section + "\n",
        encoding="utf-8",
    )

    if not args.no_readme_update:
        _update_readme(args.readme, section)

    scaling_full_chart_path = args.output_dir / "scaling_full_benchmark.png"
    scaling_subset_chart_path = args.output_dir / "scaling_subset_benchmark.png"
    multithread_chart_path = args.output_dir / "multithread_benchmark.png"
    multithread_report_path = args.output_dir / "multithread_benchmark.md"
    has_scaling_full_chart = False
    has_scaling_subset_chart = False
    has_multithread_chart = False
    if args.scaling:
        subset_fraction = min(1.0, max(0.01, args.scaling_subset_fraction))
        print(
            f"Running scaling benchmark "
            f"({args.scaling_points} points, up to {args.scaling_max_gb:.1f} GB, "
            f"{args.scaling_repeats} repeats each, "
            f"subset={subset_fraction:.0%} columns) …"
        )
        scaling_data = _bench_scaling(
            cols=args.cols,
            max_bytes=int(args.scaling_max_gb * 1024**3),
            num_points=args.scaling_points,
            repeats=args.scaling_repeats,
            seed=args.seed,
            subset_fraction=subset_fraction,
        )
        has_scaling_full_chart = _plot_scaling(
            scaling_data,
            scaling_full_chart_path,
            mode="full",
        )
        has_scaling_subset_chart = _plot_scaling(
            scaling_data,
            scaling_subset_chart_path,
            mode="subset",
        )

    if args.multithread:
        subset_fraction = min(1.0, max(0.01, args.scaling_subset_fraction))
        raw_target_mbs = _parse_target_mbs(args.multithread_target_mb)
        target_mbs = _expand_target_mbs(
            raw_target_mbs,
            max(2, args.multithread_scan_points),
        )
        print(
            "Running multithread benchmark "
            f"(targets={target_mbs}, files={args.multithread_files}, "
            f"workers={args.multithread_workers}, repeats={args.multithread_repeats}, "
            f"subset={subset_fraction:.0%} random columns) …"
        )
        multithread_results = _bench_multithread_subset_reads(
            target_mbs=target_mbs,
            cols=args.cols,
            seed=args.seed,
            subset_fraction=subset_fraction,
            repeats=args.multithread_repeats,
            files_per_size=args.multithread_files,
            workers=args.multithread_workers,
        )
        multithread_report = _build_multithread_markdown(multithread_results)
        multithread_report_path.write_text(multithread_report + "\n", encoding="utf-8")
        has_multithread_chart = _plot_multithread_results(
            multithread_results,
            multithread_chart_path,
        )

    print("Benchmark complete.")
    print(f"Report: {args.output_dir / 'serialisation_benchmark.md'}")
    if has_chart:
        print(f"Chart: {chart_path}")
    else:
        print("Chart: skipped (matplotlib missing)")
    if args.no_readme_update:
        print("README update: skipped")
    else:
        print(f"README update: {args.readme}")
    if args.scaling:
        if has_scaling_full_chart:
            print(f"Scaling chart (full read): {scaling_full_chart_path}")
        else:
            print(
                "Scaling chart (full read): skipped "
                "(matplotlib missing or insufficient data)"
            )
        if has_scaling_subset_chart:
            print(f"Scaling chart (subset read): {scaling_subset_chart_path}")
        else:
            print(
                "Scaling chart (subset read): skipped "
                "(matplotlib missing or insufficient data)"
            )
    if args.multithread:
        print(f"Multithread report: {multithread_report_path}")
        if has_multithread_chart:
            print(f"Multithread chart: {multithread_chart_path}")
        else:
            print("Multithread chart: skipped (matplotlib missing or no data)")


if __name__ == "__main__":
    main()
