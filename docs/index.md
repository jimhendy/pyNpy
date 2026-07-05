# df-npy

`df-npy` serializes and deserializes pandas DataFrames to `.npy` data files plus JSON metadata.

## Why use it

`df-npy` is built for fast data access patterns where deserialization overhead dominates throughput.

- Fast full-frame reads for homogeneous numeric and string-safe data.
- Identifier-based subset loading to avoid unnecessary column materialization.
- Explicit no-pickle policy for security-sensitive environments.

## Typical workflow

1. Serialize once:

```python
NpySerializer.to_npy(df, "dataset.npy")
```

2. Load only needed columns during compute graph execution:

```python
NpySerializer.from_npy("dataset.npy", identifiers=["col_a", "col_b"])
```

3. Run performance checks:

```bash
just perf
```

## Speed notes

⚠️ **Benchmarks are directional, not universal constants.** Results vary with dataset
shape, storage backend, filesystem cache state, and system I/O load.

**Where df-npy has a measurable edge: concurrent columnar subsetting.**

Because the on-disk layout is a plain Fortran-order NumPy array, `from_npy` with
`identifiers=` only page-faults in the columns you request. When many files are read
concurrently with a partial column selection, this translates into lower wall-time
than formats that must decode the full stream regardless of the columns requested.

Benchmarked results (8 workers, 10 files, random 50 % column subset, cold-cache reads,
median of 7 repeats — see `benchmark/multithread_benchmark.md`):

| File size | df-npy | pickle | feather | parquet |
|---:|---:|---:|---:|---:|
| 1 MB  | ~18ms | ~19ms | ~24ms | ~39ms |
| 10 MB | ~56ms | ~55ms | ~65ms | ~90ms |
| 100 MB | ~0.41s | ~0.72s | ~0.56s | ~0.66s |

At small file sizes (~1–7 MB) differences are within noise. The gap opens above
~10 MB and is clearest at 50–100 MB. For single-file full-DataFrame reads, pickle
is typically faster because it streams the entire file sequentially rather than
page-faulting through a memory map.

**Other caveats:**

- Write times have high variance (2–3× ranges are common); treat write benchmarks
  as rough order-of-magnitude only.
- Cache eviction (`posix_fadvise(POSIX_FADV_DONTNEED)`) is advisory — the kernel
  may ignore it for recently-written pages.  Cold-read measurements are best-effort.
- Compression: df-npy stores data uncompressed. Parquet and feather use compression
  by default, so their on-disk footprint is smaller even when read-time is slower.

## Security

This project enforces a strict no-pickle policy:

- Writes always use `allow_pickle=False`.
- Reads always use `allow_pickle=False`.
- Data requiring pickle-backed object serialization is rejected.

## Local docs

Run docs locally:

```bash
just docs
```

Further reading:

- [Quickstart](usage/quickstart.md)
- [Performance](usage/performance.md)
- [API Reference](api.md)
