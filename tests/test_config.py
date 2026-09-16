import pytest

from qnn.config import ExperimentConfig


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="Unknown configuration"):
        ExperimentConfig.from_mapping({"seed": 7, "epohs": 3})


def test_integer_fields_do_not_silently_truncate_floats():
    with pytest.raises(TypeError):
        ExperimentConfig.from_mapping({"num_qubits": 2.7})


def test_cli_config_requires_two_features():
    with pytest.raises(ValueError, match="exactly two"):
        ExperimentConfig(num_features=3, num_qubits=3)


def test_grad_clip_zero_is_valid_and_means_disabled():
    cfg = ExperimentConfig(grad_clip=0.0)
    assert cfg.grad_clip == 0.0


def test_experiment_defaults_match_checked_in_default_json():
    import json
    from pathlib import Path

    checked_in = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
    assert ExperimentConfig().to_dict() == checked_in


def test_experiment_config_rejects_non_mapping_input():
    with pytest.raises(TypeError, match="Mapping"):
        ExperimentConfig.from_mapping([("seed", 7)])  # type: ignore[arg-type]
