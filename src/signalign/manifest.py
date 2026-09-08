"""Target-free manifests shared by canonicalization and hand refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from signalign.io_utils import atomic_write_json, atomic_write_text, sha256_file


_DIGITS = re.compile(r"\d+")


def first_int(path: Path) -> int:
    match = _DIGITS.search(path.stem)
    if match is None:
        raise ValueError(f"filename has no integer: {path}")
    return int(match.group())


@dataclass(frozen=True)
class FrameRecord:
    """One RGB frame used without any target annotation during inference."""

    sign: str
    sign_class: str
    source_path: str
    source_frame_id: int
    sequence_index: int


@dataclass(frozen=True)
class HandFrameRecord:
    """A canonical SMPL-X state paired with a frozen hand-expert observation."""

    record_id: str
    sign: str
    sign_class: str
    frame_index: int
    source_frame_id: int
    rgb_path: str
    canonical_state_path: str
    canonical_obj_path: str
    rgb_sha256: str
    state_sha256: str
    obj_sha256: str


def read_sign_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if fields:
            if len(fields) < 2:
                raise ValueError(f"expected '<sign> <class>': {raw!r}")
            result[fields[0]] = fields[1]
    return dict(sorted(result.items()))


def image_paths(folder: Path) -> list[Path]:
    result = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    return sorted(result, key=first_int)


def write_jsonl(records: Iterable[object], path: Path) -> None:
    content = "\n".join(json.dumps(asdict(item), sort_keys=True) for item in records)
    atomic_write_text(path, content + ("\n" if content else ""))


def read_jsonl(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def read_hand_manifest(path: Path) -> list[HandFrameRecord]:
    return [
        HandFrameRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prepare_inference_manifests(
    rgb_root: Path,
    signs_file: Path,
    segments_file: Path,
    output_root: Path,
    *,
    expected_signs: int | None = None,
    expected_frames: int | None = None,
) -> dict[str, object]:
    """Create manifests from RGB and public segment metadata only.

    Ground-truth paths are deliberately absent from this API.
    """
    signs = read_sign_file(signs_file)
    segments = json.loads(segments_file.read_text(encoding="utf-8"))
    if set(signs) != set(segments):
        raise RuntimeError(f"sign/segment mismatch: {sorted(set(signs) ^ set(segments))}")
    summaries: list[dict[str, object]] = []
    total = 0
    for sign, sign_class in signs.items():
        start, end = map(int, segments[sign])
        selected = [
            path for path in image_paths(rgb_root / sign)
            if start <= first_int(path) <= end
        ]
        if not selected:
            raise RuntimeError(f"no central RGB frames for {sign}: {(start, end)}")
        records = [
            FrameRecord(
                sign=sign,
                sign_class=sign_class,
                source_path=str(path.resolve()),
                source_frame_id=first_int(path),
                sequence_index=index,
            )
            for index, path in enumerate(selected)
        ]
        manifest_path = output_root / f"{sign}.jsonl"
        write_jsonl(records, manifest_path)
        total += len(records)
        summaries.append(
            {
                "sign": sign,
                "class": sign_class,
                "frames": len(records),
                "sha256": sha256_file(manifest_path),
            }
        )
    if expected_signs is not None and len(summaries) != expected_signs:
        raise RuntimeError(f"sign count {len(summaries)} != {expected_signs}")
    if expected_frames is not None and total != expected_frames:
        raise RuntimeError(f"frame count {total} != {expected_frames}")
    report = {
        "schema_version": "signalign.inference-manifest.v1",
        "target_free": True,
        "signs": len(summaries),
        "frames": total,
        "items": summaries,
    }
    atomic_write_json(output_root / "summary.json", report)
    return report
