# Quickstart

## Install dependencies

Use the project environment and install dev dependencies:

```bash
uv sync
```

## Basic serialization

```python
from pathlib import Path

import pandas as pd

from df_npy import NpySerializer

frame = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "value": [10.0, 20.0, 30.0],
        "name": ["a", "b", "c"],
    }
)

path = Path("data/sample.npy")
NpySerializer.to_npy(frame, path)
loaded = NpySerializer.from_npy(path)
```

## Identifier subset loading

For large files, load only required columns:

```python
subset = NpySerializer.from_npy(path, identifiers=["id", "value"])
```

This is designed to reduce deserialization overhead compared to loading all columns.

## Security model

`df-npy` enforces no pickle usage:

- Writes always use `allow_pickle=False`.
- Reads always use `allow_pickle=False`.
- Data requiring pickle-backed object serialization is rejected.
