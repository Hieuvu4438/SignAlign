"""Construct a deterministic full-coverage view of frozen initializers."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from signalign.io_utils import atomic_write_json, sha256_file, tree_sha256
from signalign.manifest import read_jsonl


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _manifest_identities(manifest: Path) -> list[tuple[str, str]]:
    if manifest.is_dir():
        records = []
        for path in sorted(manifest.glob("*.jsonl")):
            records.extend(read_jsonl(path))
        return [(record.sign, Path(record.source_path).stem) for record in records]
    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(row["sign"], Path(row["prediction_path"]).stem) for row in rows]


def _manifest_sha256(manifest: Path) -> str:
    return tree_sha256(manifest, "*.jsonl") if manifest.is_dir() else sha256_file(manifest)


def build_initializer_view(
    manifest: Path,
    primary: Path,
    fallback: Path,
    output: Path,
) -> dict[str, object]:
    """Use a primary reconstruction when complete, else one whole fallback frame.

    ``manifest`` may be a public per-sign JSONL manifest directory or the
    legacy CSV contract with ``sign`` and ``prediction_path`` columns.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    identities = _manifest_identities(manifest)
    if not identities:
        raise ValueError("initializer manifest is empty")
    if len(identities) != len(set(identities)):
        raise ValueError("initializer manifest has duplicate sign/frame identities")
    counts: Counter[str] = Counter()
    selections = []
    for sign, frame in identities:
        result = Path(sign) / "smplifyx/results" / f"{frame}.pkl"
        mesh = Path(sign) / "smplifyx/meshes" / f"{frame}.obj"
        use_primary = (primary / result).is_file() and (primary / mesh).is_file()
        source = primary if use_primary else fallback
        label = "primary" if use_primary else "fallback"
        _link(source / result, output / result)
        _link(source / mesh, output / mesh)
        counts[label] += 1
        selections.append({"sign": sign, "frame": frame, "source": label})
    report = {
        "schema_version": "signalign.initializer-view.v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _manifest_sha256(manifest),
        "primary": str(primary.resolve()),
        "fallback": str(fallback.resolve()),
        "output": str(output.resolve()),
        "frames": len(identities),
        "primary_frames": counts["primary"],
        "fallback_frames": counts["fallback"],
        "fallback_fraction": counts["fallback"] / len(identities),
        "selection": selections,
    }
    atomic_write_json(output / "locked_view_manifest.json", report)
    return report
