import json
from pathlib import Path

import numpy as np
import pytest

from qnn.cli import build_parser, main, predict_command
from qnn.qnn import DataReuploadingQNN


def test_predict_infers_architecture_from_parameter_shape(tmp_path: Path, capsys):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, seed=3)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, model.params, allow_pickle=False)
    code = main(["predict", "--params", str(params_path), "--x0", "0.2", "--x1", "-0.1"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 <= payload["p_class_1"] <= 1.0
    assert payload["prediction"] in (0, 1)


def test_predict_uses_sibling_saved_model_metadata(tmp_path: Path, capsys):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, observable_wire=0, seed=3)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, model.params, allow_pickle=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"model": model.architecture()}), encoding="utf-8"
    )
    assert main(["predict", "--params", str(params_path), "--x0", "0.1", "--x1", "0.2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    expected = float(model.predict_proba(np.array([[0.1, 0.2]]) * np.pi)[0])
    np.testing.assert_allclose(payload["p_class_1"], expected)


def test_predict_rejects_architecture_assertion_mismatch(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1)
    path = tmp_path / "p.npy"
    np.save(path, model.params, allow_pickle=False)
    with pytest.raises(SystemExit):
        main([
            "predict", "--params", str(path), "--x0", "0", "--x1", "0", "--num-layers", "2"
        ])


def test_train_rejects_unknown_config_key(tmp_path: Path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"epohs": 2}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["train", "--config", str(config), "--no-plots"])


def test_predict_rejects_noninteger_saved_architecture_metadata(tmp_path: Path):
    params = np.zeros((1, 1, 3), dtype=np.float64)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, params, allow_pickle=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {"num_layers": 1.0, "num_qubits": 1, "num_features": 2}}),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["predict", "--params", str(params_path), "--x0", "0", "--x1", "0"])
    with pytest.raises(TypeError, match="must be an integer"):
        predict_command(args)


def test_predict_rejects_saved_model_with_more_than_two_features(tmp_path: Path):
    params = np.zeros((1, 2, 3), dtype=np.float64)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, params)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "num_layers": 1,
                    "num_qubits": 2,
                    "num_features": 3,
                    "observable_wire": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["predict", "--params", str(params_path), "--x0", "0.1", "--x1", "0.2"])


def test_predict_observable_wire_is_assertion_when_metadata_exists(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, observable_wire=1, seed=3)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, model.params)
    (tmp_path / "config.json").write_text(
        json.dumps({"model": model.architecture()}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--params",
                str(params_path),
                "--x0",
                "0.1",
                "--x1",
                "0.2",
                "--observable-wire",
                "0",
            ]
        )


def test_predict_rejects_wrong_saved_parameter_count(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, seed=3)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, model.params)
    metadata = model.architecture()
    metadata["parameter_count"] = metadata["parameter_count"] + 1
    (tmp_path / "config.json").write_text(
        json.dumps({"model": metadata}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        main(["predict", "--params", str(params_path), "--x0", "0.1", "--x1", "0.2"])


def test_predict_rejects_unrecognized_config_instead_of_ignoring_it(tmp_path: Path):
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, seed=3)
    params_path = tmp_path / "trained_params.npy"
    np.save(params_path, model.params, allow_pickle=False)
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--params",
                str(params_path),
                "--config",
                str(config_path),
                "--x0",
                "0.1",
                "--x1",
                "0.2",
            ]
        )


def test_train_command_end_to_end_without_plots(tmp_path: Path, capsys):
    output_dir = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 2,
                "num_samples": 12,
                "test_size": 0.25,
                "dataset": "vertical",
                "noise": 0.01,
                "num_qubits": 1,
                "num_layers": 1,
                "num_features": 2,
                "observable_wire": 0,
                "init_scale": 0.1,
                "epochs": 1,
                "learning_rate": 0.02,
                "batch_size": 0,
                "grad_clip": 2.0,
                "output_dir": str(output_dir),
                "log_every": 1,
            }
        ),
        encoding="utf-8",
    )
    code = main(["train", "--config", str(config), "--no-plots"])
    assert code == 0
    assert (output_dir / "trained_params.npy").is_file()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert not (output_dir / "decision_boundary.png").exists()
    captured = capsys.readouterr().out
    assert "Training complete" in captured
