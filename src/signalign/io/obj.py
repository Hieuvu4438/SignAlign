from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.shape != (10475, 3):
        raise ValueError(f"vertices shape {vertices.shape}")
    if faces.shape != (20908, 3):
        raise ValueError(f"faces shape {faces.shape}")
    if not np.isfinite(vertices).all():
        raise FloatingPointError("mesh contains NaN/Inf")
    if faces.min() < 0 or faces.max() >= 10475:
        raise IndexError("face index outside vertex range")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    validate_mesh(vertices, faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces + 1:
            handle.write(f"f {a} {b} {c}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
