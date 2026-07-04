"""Public package API for df_npy."""

from pathlib import Path

from ._serializer import NpySerializer


def get_columns(file_path: Path | str) -> set[str]:
    """Return serialized DataFrame columns as a set of strings."""
    return NpySerializer.get_columns(file_path)


__all__ = ["NpySerializer", "get_columns"]
