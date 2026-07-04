# df-npy

Serialize and deserialize pandas DataFrames to `.npy` plus JSON metadata.

## Security

### Pickle policy

`df-npy` has a strict no-pickle policy.

- Serialization always writes NumPy arrays with `allow_pickle=False`.
- Deserialization always reads with `allow_pickle=False`.
- DataFrames that would require pickle-backed object serialization are rejected.

This behavior is intentional for security-sensitive environments.

### What this means in practice

Supported data should be representable without Python object pickling.

Examples of unsupported frames:

- Object/string frames containing non-string Python objects.
- Heterogeneous object columns that rely on pickle to round-trip values.

When unsupported data is provided, `df-npy` raises `ValueError` rather than falling back to pickle.

## API

Public API:

- `NpySerializer.to_npy(df, file_path)`
- `NpySerializer.from_npy(file_path, identifiers=None)`

Import:

```python
from df_npy import NpySerializer
```
