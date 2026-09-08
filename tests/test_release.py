from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from signalign.io_utils import tree_sha256


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_third_party_lock_uses_immutable_revisions() -> None:
    payload = json.loads((ROOT / "third_party.lock.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "signalign.third-party-lock.v1"
    assert set(payload["dependencies"]) == {"DexAvatar", "WiLoR"}
    for dependency in payload["dependencies"].values():
        assert dependency["repository"].startswith("https://github.com/")
        commit = dependency["commit"]
        assert len(commit) == 40
        assert all(character in "0123456789abcdef" for character in commit)


def test_release_has_no_private_or_generated_tree() -> None:
    forbidden = {
        "_archive",
        "archive",
        "experiment",
        "experiments",
        "inputs",
        "outputs",
        "reference",
    }
    published = {
        path.name.lower()
        for path in ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert not (forbidden & published)
    assert (ROOT / "assets" / "method_overview.png").stat().st_size > 0


def test_gitignore_covers_external_assets_and_outputs() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    patterns = {line.strip() for line in content}
    assert {"external/", "checkpoints/", "data/", "outputs/", "*.pkl", "*.obj"} <= patterns


def test_reference_run_supports_readme_metrics() -> None:
    payload = json.loads((ROOT / "docs" / "reference_run.json").read_text())
    assert payload["schema_version"] == "signalign.reference-run.v1"
    assert payload["split"] == {"frames": 1493, "signs": 57}
    assert payload["inference_audit"]["decision"] == "PASS"
    assert payload["inference_audit"]["source_tree_sha256"] == tree_sha256(
        ROOT / "src" / "signalign"
    )
    assert payload["evaluation"]["predictions_frozen_before_evaluation"] is True


def test_frontend_plan_uses_wilor_and_selected_fitting_config(tmp_path: Path) -> None:
    runner = load_script("run_dexavatar_wilor.py")
    image_dir = tmp_path / "rgb" / "sign_001"
    output_dir = tmp_path / "output" / "sign_001"
    dexavatar = tmp_path / "DexAvatar"
    wilor = tmp_path / "WiLoR"
    fitting_root = dexavatar / "dexavatar_fitting"
    fitting_config = fitting_root / "cfg_files" / "fit_smplx_vposer_x.yaml"

    steps = runner.build_steps(
        image_dir=image_dir,
        output_dir=output_dir,
        dexavatar=dexavatar,
        wilor=wilor,
        fitting_root=fitting_root,
        fitting_config=fitting_config,
        sapiens_env="sapiens",
        smplerx_env="smplerx",
        wilor_env="wilor",
        fitting_env="fitting",
        fast_wilor=False,
    )

    assert [step.name for step in steps] == [
        "sapiens",
        "aggregate-sapiens",
        "smpler-x",
        "mean-shape",
        "wilor",
        "dexavatar-fit",
    ]
    wilor_step = next(step for step in steps if step.name == "wilor")
    assert str(wilor) in wilor_step.command
    assert "export_wilor_for_dexavatar.py" in " ".join(wilor_step.command)
    fitting_step = next(step for step in steps if step.name == "dexavatar-fit")
    config_index = fitting_step.command.index("--config")
    assert fitting_step.command[config_index + 1] == str(fitting_config)
    assert fitting_step.cwd == fitting_root


def test_dexavatar_exporter_uses_public_record_identity() -> None:
    exporter = load_script("export_wilor_for_dexavatar.py")
    assert exporter.public_record_key("low_0164.png", "/data/sign_001") == "sign_001/164"
    assert exporter.public_record_key("low_0164.png", None) == "low_0164.png"
