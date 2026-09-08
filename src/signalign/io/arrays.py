from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ without ever exposing a partial cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
