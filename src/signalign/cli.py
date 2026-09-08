"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signalign.evaluation import evaluate_official
from signalign.export import export_evaluation_layout
from signalign.frontend import (
    build_initializer_view,
    build_wilor_frame_manifest,
    import_wilor_sidecar,
    merge_wilor_sidecars,
    validate_wilor_cache,
)
from signalign.manifest import prepare_inference_manifests
from signalign.pipeline import run_inference


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="signalign")
    commands = root.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("infer", help="run target-free inference")
    inference.add_argument("--config", type=Path, required=True)

    manifests = commands.add_parser(
        "prepare-manifests", help="create target-free per-sign RGB manifests"
    )
    manifests.add_argument("--rgb-root", type=Path, required=True)
    manifests.add_argument("--signs", type=Path, required=True)
    manifests.add_argument("--segments", type=Path, required=True)
    manifests.add_argument("--output", type=Path, required=True)
    manifests.add_argument("--expected-signs", type=int)
    manifests.add_argument("--expected-frames", type=int)

    initializer = commands.add_parser(
        "build-initializer", help="build a deterministic primary/fallback view"
    )
    initializer.add_argument("--manifest", type=Path, required=True)
    initializer.add_argument("--primary", type=Path, required=True)
    initializer.add_argument("--fallback", type=Path, required=True)
    initializer.add_argument("--output", type=Path, required=True)

    wilor_manifest = commands.add_parser(
        "prepare-wilor", help="freeze RGB metadata for WiLoR extraction"
    )
    wilor_manifest.add_argument("--manifests", type=Path, required=True)
    wilor_manifest.add_argument("--output", type=Path, required=True)

    wilor_import = commands.add_parser(
        "import-wilor", help="convert a raw WiLoR sidecar into inference inputs"
    )
    wilor_import.add_argument("--manifests", type=Path, required=True)
    wilor_import.add_argument("--sidecar", type=Path, required=True)
    wilor_import.add_argument("--output", type=Path, required=True)

    wilor_merge = commands.add_parser(
        "merge-wilor", help="merge per-sign raw WiLoR sidecars"
    )
    wilor_merge.add_argument("--sidecars", type=Path, nargs="+", required=True)
    wilor_merge.add_argument("--output", type=Path, required=True)

    wilor_validate = commands.add_parser(
        "validate-wilor", help="validate all cached hand observations"
    )
    wilor_validate.add_argument("--manifests", type=Path, required=True)
    wilor_validate.add_argument("--cache", type=Path, required=True)
    wilor_validate.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export", help="freeze evaluator-layout meshes")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    evaluation = commands.add_parser("evaluate", help="evaluate frozen predictions")
    evaluation.add_argument("--evaluator", type=Path, required=True)
    evaluation.add_argument("--evaluator-sha256", required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--reference", type=Path, required=True)
    evaluation.add_argument("--signs", type=Path, required=True)
    evaluation.add_argument("--segments", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--method", default="SignAlign")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "infer":
        result = run_inference(args.config)
    elif args.command == "prepare-manifests":
        result = prepare_inference_manifests(
            args.rgb_root,
            args.signs,
            args.segments,
            args.output,
            expected_signs=args.expected_signs,
            expected_frames=args.expected_frames,
        )
    elif args.command == "build-initializer":
        result = build_initializer_view(
            args.manifest, args.primary, args.fallback, args.output
        )
    elif args.command == "prepare-wilor":
        result = build_wilor_frame_manifest(args.manifests, args.output)
    elif args.command == "import-wilor":
        result = import_wilor_sidecar(args.manifests, args.sidecar, args.output)
    elif args.command == "merge-wilor":
        result = merge_wilor_sidecars(args.sidecars, args.output)
    elif args.command == "validate-wilor":
        result = validate_wilor_cache(args.manifests, args.cache, args.output)
    elif args.command == "export":
        result = export_evaluation_layout(args.manifest, args.predictions, args.output)
    elif args.command == "evaluate":
        result = evaluate_official(
            args.evaluator,
            args.evaluator_sha256,
            args.predictions,
            args.reference,
            args.signs,
            args.segments,
            args.output,
            method=args.method,
        )
    else:
        raise AssertionError(args.command)
    printable = dict(result)
    for large_field in ("selection", "records", "image_sha256", "items"):
        if large_field in printable:
            printable[f"{large_field}_count"] = len(printable[large_field])
            del printable[large_field]
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
