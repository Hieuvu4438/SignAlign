"""Official evaluation isolated from the inference pipeline."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from signalign.io_utils import atomic_write_json, atomic_write_text, sha256_file

METRIC = re.compile(
    r"\[(?P<method>[^\]]+)\]:\s+(?P<metric>[^:]+):\s+(?P<value>[0-9.+-eE]+)\s+\(mm\)"
)


def parse_metrics(text: str) -> dict[str, float]:
    values = {
        " ".join(match.group("metric").lower().split()): float(match.group("value"))
        for match in METRIC.finditer(text)
    }
    required = {
        "tr all",
        "tr above pelvis upper body",
        "tr above pelvis minus face",
        "tr above pelvis minus head",
        "tr left hand",
        "tr right hand",
    }
    if missing := required - set(values):
        raise ValueError(f"official metrics missing: {sorted(missing)}")
    return values


def evaluate_official(
    evaluator: Path,
    evaluator_sha256: str,
    prediction_root: Path,
    reference_root: Path,
    signs_file: Path,
    segments_file: Path,
    output_root: Path,
    method: str = "SignAlign",
    python: str = sys.executable,
) -> dict[str, object]:
    """Evaluate already-frozen meshes; this function is never imported by pipeline.py."""
    digest = sha256_file(evaluator)
    if digest != evaluator_sha256:
        raise RuntimeError(f"official evaluator checksum mismatch: {digest}")
    command = [
        python,
        str(evaluator),
        "--method",
        method,
        "--central",
        "true",
        "--evaluate_folder",
        str(prediction_root),
        "--gt_folder",
        str(reference_root),
        "--sign_file",
        str(signs_file),
        "--sign_seg",
        str(segments_file),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    process = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_root / "official_stdout.txt", process.stdout)
    atomic_write_text(output_root / "official_stderr.txt", process.stderr)
    report: dict[str, object] = {
        "schema_version": "signalign.official-evaluation.v1",
        "command": command,
        "evaluator_sha256": digest,
        "returncode": process.returncode,
        "predictions_frozen_before_evaluation": True,
    }
    if process.returncode == 0:
        report["metrics_mm"] = parse_metrics(process.stdout + "\n" + process.stderr)
    atomic_write_json(output_root / "official_result.json", report)
    if process.returncode:
        raise RuntimeError(f"official evaluator failed; inspect {output_root}")
    return report
