from pathlib import Path

import pytest

from signalign.pipeline import validate_inference_config


def minimal_paths(tmp_path: Path) -> dict[str, str]:
    names = (
        "rgb_root",
        "signs_file",
        "segments_file",
        "initializer_root",
        "smplx_model_root",
        "mano_smplx_ids",
        "wilor_root",
    )
    result = {}
    for name in names:
        path = tmp_path / name
        path.touch()
        result[name] = str(path)
    result["output_root"] = str(tmp_path / "output")
    return result


@pytest.mark.parametrize("name", ["gt_root", "ground_truth_root", "evaluator"])
def test_evaluation_paths_are_rejected(tmp_path: Path, name: str) -> None:
    paths = minimal_paths(tmp_path)
    paths[name] = str(tmp_path / name)
    with pytest.raises(ValueError, match="forbidden"):
        validate_inference_config({"paths": paths})


def test_target_free_path_set_is_accepted(tmp_path: Path) -> None:
    validate_inference_config({"paths": minimal_paths(tmp_path)})
