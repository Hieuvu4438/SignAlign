#!/usr/bin/env python3
"""Materialize the exact third-party source revisions used by SignAlign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "third_party.lock.json"


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_lock(path: Path = LOCK_FILE) -> dict[str, dict[str, str]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "signalign.third-party-lock.v1":
        raise RuntimeError(f"unsupported lock schema: {path}")
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise RuntimeError(f"empty dependency lock: {path}")
    return dependencies


def verify_checkout(destination: Path, expected_commit: str) -> None:
    if not (destination / ".git").exists():
        raise FileNotFoundError(f"missing Git checkout: {destination}")
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual != expected_commit:
        raise RuntimeError(
            f"revision mismatch for {destination.name}: {actual} != {expected_commit}"
        )
    dirty = run(["git", "status", "--short"], cwd=destination)
    if dirty:
        raise RuntimeError(f"third-party checkout is dirty: {destination}\n{dirty}")


def materialize(name: str, spec: dict[str, str], destination: Path) -> None:
    if destination.exists():
        verify_checkout(destination, spec["commit"])
        print(f"verified {name}: {spec['commit']}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", str(destination)])
    run(["git", "remote", "add", "origin", spec["repository"]], cwd=destination)
    run(
        ["git", "fetch", "--depth", "1", "origin", spec["commit"]],
        cwd=destination,
    )
    run(["git", "checkout", "--detach", spec["commit"]], cwd=destination)
    if name == "DexAvatar":
        # Sapiens is the only submodule at the locked DexAvatar revision.
        run(
            ["git", "submodule", "update", "--init", "--depth", "1", "sapiens"],
            cwd=destination,
        )
    verify_checkout(destination, spec["commit"])
    print(f"installed {name}: {spec['commit']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or verify pinned DexAvatar and WiLoR source trees"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "external",
        help="destination directory (default: ./external)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing checkouts without fetching",
    )
    args = parser.parse_args()

    dependencies = load_lock()
    for name, spec in dependencies.items():
        destination = args.root.resolve() / name
        if args.verify_only:
            verify_checkout(destination, spec["commit"])
            print(f"verified {name}: {spec['commit']}")
        else:
            materialize(name, spec, destination)


if __name__ == "__main__":
    main()
