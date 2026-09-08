import csv
from pathlib import Path

from signalign.frontend.initializer import build_initializer_view


def write_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sign", "prediction_path"))
        writer.writeheader()
        writer.writerow({"sign": "A", "prediction_path": "low_1.obj"})
        writer.writerow({"sign": "A", "prediction_path": "low_2.obj"})


def artifact(root: Path, frame: str, content: str) -> None:
    result = root / "A/smplifyx/results" / f"{frame}.pkl"
    mesh = root / "A/smplifyx/meshes" / f"{frame}.obj"
    result.parent.mkdir(parents=True, exist_ok=True)
    mesh.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(content + "-result", encoding="utf-8")
    mesh.write_text(content + "-mesh", encoding="utf-8")


def test_initializer_view_selects_whole_frames(tmp_path: Path) -> None:
    manifest = tmp_path / "frames.csv"
    write_manifest(manifest)
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    artifact(primary, "low_1", "primary")
    artifact(fallback, "low_1", "fallback")
    artifact(fallback, "low_2", "fallback")
    output = tmp_path / "view"
    report = build_initializer_view(manifest, primary, fallback, output)
    assert report["primary_frames"] == 1
    assert report["fallback_frames"] == 1
    assert (output / "A/smplifyx/results/low_1.pkl").read_text() == "primary-result"
    assert (output / "A/smplifyx/meshes/low_1.obj").read_text() == "primary-mesh"
    assert (output / "A/smplifyx/results/low_2.pkl").read_text() == "fallback-result"
    assert (output / "A/smplifyx/meshes/low_2.obj").read_text() == "fallback-mesh"
