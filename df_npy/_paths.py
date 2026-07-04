from __future__ import annotations

from pathlib import Path

from ._constants import JSON_SUFFIX, NPY_SUFFIX


def ensure_npy_path(file_path: Path | str, *, for_write: bool) -> Path:
    path = Path(file_path)
    if path.suffix != NPY_SUFFIX:
        path = path.with_suffix(NPY_SUFFIX)
    if for_write:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def metadata_file_from_npy_file(npy_file: Path) -> Path:
    return npy_file.with_suffix(JSON_SUFFIX)
