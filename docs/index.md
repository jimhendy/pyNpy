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
