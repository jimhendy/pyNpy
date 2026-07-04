# Performance

## Why this exists

This project targets a real bottleneck: loading many parquet or feather files in concurrent graphs can become deserialization-bound. The `.npy` path is optimized for fast loading and selective column reads.

## Performance guarantees tested in CI

The integration suite validates:

1. Subset loading is faster than full loading for large `.npy` files.
2. Thread-pool subset loading across multiple files is faster than loading full datasets.
3. Optional cross-format timing comparison (`.npy` vs parquet/feather subsets) stays in a comparable range.

See [tests/test_integration_performance.py](../../tests/test_integration_performance.py).

## Run performance tests

Run all performance integration tests:

```bash
just perf
```

Run only format comparison benchmark:

```bash
just perf-formats
```

## Notes on interpreting timings

- Performance tests are relative, not absolute. They compare subset vs full behavior in the same environment.
- The parquet/feather test is optional and skipped when `pyarrow` is unavailable.
- If your workload is highly concurrent, prioritize subset reads and avoid unnecessary full-frame deserialization.
