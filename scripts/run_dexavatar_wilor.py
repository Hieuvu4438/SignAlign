#!/usr/bin/env python3
"""Run the pinned DexAvatar initializer with WiLoR hand observations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str]


def conda(env_name: str, *command: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, *command]


def validate_checkout(path: Path, required: tuple[str, ...]) -> Path:
    resolved = path.resolve()
    missing = [item for item in required if not (resolved / item).exists()]
    if missing:
        raise FileNotFoundError(f"invalid checkout {resolved}; missing {missing}")
    return resolved


def build_steps(
    *,
    image_dir: Path,
    output_dir: Path,
    dexavatar: Path,
    wilor: Path,
    fitting_root: Path,
    fitting_config: Path,
    sapiens_env: str,
    smplerx_env: str,
    wilor_env: str,
    fitting_env: str,
    fast_wilor: bool,
) -> list[Step]:
    base_env = {
        "ROOT_PATH": str(image_dir),
        "OUTPUT_PATH": str(output_dir),
    }
    python = "python"
    exporter = ROOT / "scripts" / "export_wilor_for_dexavatar.py"
    wilor_command = conda(
        wilor_env,
        python,
        str(exporter),
        "--repo",
        str(wilor),
        "--img_folder",
        str(image_dir),
        "--out_folder",
        str(output_dir),
    )
    if fast_wilor:
        wilor_command.append("--fast")
    fitting_pythonpath = os.pathsep.join([str(fitting_root), str(fitting_root / "smplifyx")])
    return [
        Step(
            "sapiens",
            conda(sapiens_env, "bash", str(dexavatar / "scripts/S1_sapiens_extract.sh")),
            dexavatar,
            base_env,
        ),
        Step(
            "aggregate-sapiens",
            conda(
                fitting_env,
                python,
                str(dexavatar / "scripts/aggregate_sapiens.py"),
                "--sapiens_dir",
                str(output_dir / "sapiens_1b"),
                "--output_path",
                str(output_dir),
                "--subfolder",
                image_dir.name,
            ),
            dexavatar,
            base_env,
        ),
        Step(
            "smpler-x",
            conda(smplerx_env, "bash", str(dexavatar / "scripts/S1_smplerx_extract.sh")),
            dexavatar,
            base_env,
        ),
        Step(
            "mean-shape",
            conda(
                fitting_env,
                python,
                str(dexavatar / "scripts/M3_mean_shape_smplerx.py"),
                "--input_path",
                str(image_dir),
                "--output_path",
                str(output_dir),
            ),
            dexavatar,
            base_env,
        ),
        Step("wilor", wilor_command, ROOT, base_env),
        Step(
            "dexavatar-fit",
            conda(
                fitting_env,
                python,
                "script.py",
                "--config",
                str(fitting_config),
                "--path",
                str(image_dir),
                "--out_path",
                str(output_dir),
                "--gpu_id",
                "0",
                "--split_num",
                "1",
            ),
            fitting_root,
            {**base_env, "PYTHONPATH": fitting_pythonpath},
        ),
    ]


def step_complete(step: Step, output_dir: Path) -> bool:
    markers = {
        "sapiens": output_dir / "sapiens_1b",
        "aggregate-sapiens": output_dir / "sapiens.pkl",
        "smpler-x": output_dir / "smplerx" / "smplx",
        "mean-shape": output_dir / "mean_shape_smplerx.npy",
        "wilor": output_dir / "hamer" / "hamer.pkl",
        "dexavatar-fit": output_dir / "smplifyx" / "results",
    }
    marker = markers[step.name]
    return marker.is_dir() and any(marker.iterdir()) if marker.is_dir() else marker.is_file()


def printable(step: Step) -> dict[str, object]:
    return {
        **asdict(step),
        "command": shlex.join(step.command),
        "cwd": str(step.cwd),
    }


def run_step(step: Step) -> None:
    environment = os.environ.copy()
    environment.update(step.env)
    print(f"[{step.name}] {shlex.join(step.command)}", flush=True)
    subprocess.run(step.command, cwd=step.cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Sapiens -> SMPLer-X -> WiLoR -> DexAvatar with the released "
            "SignBPoser/SignHPoser fitting configuration"
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=ROOT / "external")
    parser.add_argument("--fitting-config", type=Path)
    parser.add_argument("--sapiens-env", default="sapiens_fix")
    parser.add_argument("--smplerx-env", default="smpler_x")
    parser.add_argument("--wilor-env", default="wilor")
    parser.add_argument("--fitting-env", default="dexavatar")
    parser.add_argument("--fast-wilor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        parser.error(f"input root does not exist: {input_root}")
    external = args.external_root.resolve()
    dexavatar = validate_checkout(
        external / "DexAvatar",
        ("sapiens", "SMPLer-X", "dexavatar_fitting", "scripts"),
    )
    wilor = validate_checkout(external / "WiLoR", ("wilor", "pretrained_models"))
    fitting_root = dexavatar / "dexavatar_fitting"
    fitting_config = (
        args.fitting_config
        or fitting_root / "cfg_files" / "fit_smplx_vposer_x.yaml"
    ).resolve()
    if not fitting_config.is_file():
        raise FileNotFoundError(f"missing DexAvatar fitting config: {fitting_config}")

    sign_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not sign_dirs:
        parser.error(f"no sign directories under: {input_root}")
    plan: list[dict[str, object]] = []
    for image_dir in sign_dirs:
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg"}
        ]
        if not images:
            raise RuntimeError(f"no PNG/JPEG frames in sign directory: {image_dir}")
        output_dir = args.output_root.resolve() / image_dir.name
        steps = build_steps(
            image_dir=image_dir,
            output_dir=output_dir,
            dexavatar=dexavatar,
            wilor=wilor,
            fitting_root=fitting_root,
            fitting_config=fitting_config,
            sapiens_env=args.sapiens_env,
            smplerx_env=args.smplerx_env,
            wilor_env=args.wilor_env,
            fitting_env=args.fitting_env,
            fast_wilor=args.fast_wilor,
        )
        if args.dry_run:
            plan.extend(printable(step) for step in steps)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gender.txt").write_text("neutral\n", encoding="utf-8")
        for step in steps:
            if args.resume and step_complete(step, output_dir):
                print(f"[{image_dir.name}/{step.name}] already complete", flush=True)
                continue
            run_step(step)
            if not step_complete(step, output_dir):
                raise RuntimeError(
                    f"{image_dir.name}/{step.name} exited without its completion artifact"
                )
    if args.dry_run:
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        print()


if __name__ == "__main__":
    main()
